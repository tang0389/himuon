import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)

from torch.utils.data import DataLoader
from transformers import get_wsd_schedule
from datasets import load_dataset
import os
import argparse
import json
import time
from datetime import datetime

from himuon.dataset import ToyDataset
from himuon.fsdp_bank import (
    Llama3BankScheme,
    Qwen3BankScheme,
    release_pre_shard_refs,
    unbank_state_dict,
    wrap_with_banks,
)
from himuon.model import get_model_and_tokenizer, count_model_parameters, count_num_training_tokens
from himuon.optim import get_optimizer
from himuon.utils import TimeBudget, seed_everything, merge_kv_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- Model & Dataset ---
    parser.add_argument("--model", type=str, default="Qwen3-0.6B", help="Model name")
    parser.add_argument("-d", "--dataset", type=str, default="sample-10BT", help="Subset name of FineWeb dataset")
    # --- Hyperparameters ---
    parser.add_argument("--optimizer", "--opt", type=str, default="adamw", help="Optimizer name")
    parser.add_argument("--learning-rate", "--lr", dest="lr", type=float, help="Learning rate")
    parser.add_argument("--weight-decay", "--wd", type=float, help="Weight decay")
    # --- Training Configuration ---
    parser.add_argument("--batch-size", "--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", "--grad-acc", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--num-training-steps", "--steps", type=int, help="Number of training steps")
    parser.add_argument("--num-warmup-steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--num-decay-steps", type=int, default=400, help="WSD decay phase length (peak -> min_lr_ratio*peak)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="WSD plateau LR as fraction of peak")
    parser.add_argument("--reconfigure-schedule", type=str, default="", help='JSON list of dicts with "step" key; other keys forwarded to optimizer.reconfigure().')
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False, help="CUDA-graph capture for HiMuon (default OFF under FSDP2). Controls zero_grad(set_to_none).")
    parser.add_argument("--time-stats", action="store_true", help="Log per-step wall-clock time + tokens/sec on rank 0 (synchronizes rank-0 CUDA each step).")
    parser.add_argument("--time-budget", type=float, default=None, help="Stop training after X hours if provided")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload")
    # --- Evaluation ---
    parser.add_argument("--eval-interval", type=int, default=0, help="Run held-out eval every N steps (0=disabled)")
    parser.add_argument("--eval-steps", type=int, default=10, help="Number of batches per eval")
    parser.add_argument("--eval-dataset", type=str, default="CC-MAIN-2024-51", help="FineWeb subset for held-out eval (should not overlap with training subset)")
    # --- System & Logging ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save checkpoint")
    parser.add_argument("--run-name", type=str, default=None, help="Override WandB run name")
    parser.add_argument("--kwargs", nargs="*", default=[], help="Extra key=val args, e.g. --kwargs ns_steps=10 nestrov=False")
    args = parser.parse_args()
    merge_kv_args(args)

    seed_everything(args.seed)

    # setup FSDP2
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 1D data-parallel mesh. HSDP (2D) would replace this with
    # ``init_device_mesh("cuda", (num_nodes, gpus_per_node), ...)``.
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    if rank == 0:
        # setup logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        run_name = args.run_name or f"{args.model}_fsdp2_{args.optimizer}_lr{args.lr}_{timestamp}"
        if args.wandb:
            from himuon.logger import WandbLogger
            logger = WandbLogger(args, run_name)
        else:
            from himuon.logger import LoguruLogger
            logger = LoguruLogger(run_name)
        logger.info(args)
        logger.info(f"Starting FSDP2 training with {world_size} GPUs")

    # load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.to(device=device, dtype=torch.bfloat16)

    # HiMuon requires parameter-bank sharding (tiles must stay shard-local).
    use_banks = args.optimizer.lower() in ("himuon", "muon-fsdp")
    if use_banks:
        schemes = {"qwen3": Qwen3BankScheme, "llama": Llama3BankScheme}
        model_type = model.config.model_type
        if model_type not in schemes:
            raise ValueError(
                f"{args.optimizer} under FSDP requires a BankScheme; supported "
                f"model_types: {list(schemes)}. Got {model_type!r}."
            )
        model, banks = wrap_with_banks(model, schemes[model_type](), world_size=world_size)

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    offload_policy = CPUOffloadPolicy() if args.cpu_offload else OffloadPolicy()

    if use_banks:
        fully_shard(model, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)
        release_pre_shard_refs(banks)
        torch.cuda.empty_cache()
        shard_topology = (
            f"bank-sharded ({model_type}, {len(banks)} banks) on mesh={tuple(mesh.shape)}"
        )
    else:
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)
        shard_topology = f"per-layer-sharded ({len(model.model.layers)} layers + root) on mesh={tuple(mesh.shape)}"

    if rank == 0:
        logger.info(f"FSDP2: {shard_topology}")
        logger.info(f"Mixed precision: param={mp_policy.param_dtype}, reduce={mp_policy.reduce_dtype}")
        logger.info(count_model_parameters(model)[1])

    # load streaming dataset, shuffle with buffer
    dataset = load_dataset("HuggingFaceFW/fineweb", name=args.dataset, split="train", streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=10000)
    dataset = dataset.shard(num_shards=world_size, index=rank)
    toy_dataset = ToyDataset(dataset, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(toy_dataset, batch_size=args.batch_size, num_workers=0)

    # load eval dataset (different FineWeb subset for true held-out validation)
    # Pre-collect fixed batches so every eval pass uses identical data
    if args.eval_interval > 0:
        eval_dataset = load_dataset("HuggingFaceFW/fineweb", name=args.eval_dataset, split="train", streaming=True)
        eval_dataset = eval_dataset.shuffle(seed=args.seed, buffer_size=10000)
        eval_dataset = eval_dataset.shard(num_shards=world_size, index=rank)
        eval_toy_dataset = ToyDataset(eval_dataset, tokenizer, max_length=args.max_length)
        _eval_iter = iter(DataLoader(eval_toy_dataset, batch_size=args.batch_size, num_workers=0))
        eval_batches = [next(_eval_iter) for _ in range(args.eval_steps)]
        del _eval_iter

    # load optimizer and scheduler
    if args.num_training_steps is None:
        num_training_tokens = count_num_training_tokens(model)
        tokens_per_step = args.batch_size * args.max_length * args.gradient_accumulation_steps * world_size
        args.num_training_steps = int(num_training_tokens / tokens_per_step)
    if rank == 0:
        logger.info(f"Number of training steps: {args.num_training_steps}")
    optimizer = get_optimizer(args.optimizer, model, args)
    if rank == 0:
        logger.log_optimizer_source(optimizer)
    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_stable_steps=0,
        num_decay_steps=args.num_decay_steps,
        num_training_steps=args.num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
        decay_type="cosine",
        warmup_type="linear",
    )

    reconfigure_schedule = json.loads(args.reconfigure_schedule) if args.reconfigure_schedule else []
    for _e in reconfigure_schedule:
        assert isinstance(_e, dict) and "step" in _e, f"reconfigure schedule entry must be a dict with 'step' key, got {_e!r}"

    def apply_reconfigure(at_step):
        if not reconfigure_schedule or not hasattr(optimizer, "reconfigure"):
            return
        for entry in reconfigure_schedule:
            if entry["step"] != at_step:
                continue
            kwargs = {k: v for k, v in entry.items() if k != "step"}
            if not kwargs:
                continue
            old = {k: next((g[k] for g in optimizer.param_groups if k in g), None) for k in kwargs}
            optimizer.reconfigure(**kwargs)
            new = {k: next((g[k] for g in optimizer.param_groups if k in g), None) for k in kwargs}
            if rank == 0:
                diff = ", ".join(f"{k} {old[k]} -> {new[k]}" for k in kwargs if old[k] != new[k])
                logger.info(f"step {at_step}, reconfigure: {diff or 'no change'}")

    apply_reconfigure(0)

    # train
    time_budget = TimeBudget(args.time_budget)
    if args.time_budget and rank == 0:
        logger.info(f"Training time budget remaining: {time_budget}")
    model.train()
    global_step = 0
    n_tokens_local = torch.tensor(0, dtype=torch.long, device=device)
    acc_loss = torch.tensor(0.0, device=device)
    n_tokens_prev = 0
    t_step_start = time.time() if args.time_stats and rank == 0 else None
    for step, batch in enumerate(dataloader):
        n_tokens_local += (batch['labels'] != tokenizer.eos_token_id).sum()
        batch = {k: v.to(device) for k, v in batch.items()}
        is_sync_step = (step + 1) % args.gradient_accumulation_steps == 0
        # FSDP2 grad-accum: skip reduce-scatter on non-sync steps
        model.set_requires_gradient_sync(is_sync_step)
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        acc_loss += loss.detach()

        if is_sync_step:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=not args.cuda_graph)
            global_step += 1
            apply_reconfigure(global_step)
            if t_step_start is not None:
                torch.cuda.synchronize()
                step_time = time.time() - t_step_start
            should_stop = False

            # Aggregate tokens across all ranks
            n_tokens_global = n_tokens_local.clone()
            dist.all_reduce(n_tokens_global, op=dist.ReduceOp.SUM)

            # Held-out eval (all ranks must participate for FSDP forward)
            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                model.eval()
                eval_loss = torch.tensor(0.0, device=device)
                with torch.no_grad():
                    for eval_batch in eval_batches:
                        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                        eval_outputs = model(**eval_batch)
                        eval_loss += eval_outputs.loss.detach()
                eval_loss_val = (eval_loss / args.eval_steps).item()
                model.train()
            else:
                eval_loss_val = None

            if rank == 0:
                metrics = {
                    "step": global_step,
                    "lr": optimizer.param_groups[0]['lr'],
                    "loss": acc_loss.item(),
                    "tokens": n_tokens_global.item(),
                }
                if eval_loss_val is not None:
                    metrics["eval_loss"] = eval_loss_val
                if t_step_start is not None:
                    metrics["step_time"] = step_time
                    metrics["tokens_per_sec"] = (n_tokens_global.item() - n_tokens_prev) / step_time
                    n_tokens_prev = n_tokens_global.item()
                logger.log_metrics(metrics)
                if global_step >= args.num_training_steps:
                    should_stop = True
                    stop_reason = "Max training steps reached."
                if time_budget.is_expired():
                    should_stop = True
                    stop_reason = "Time budget exceeded."
                if should_stop:
                    logger.info(f"Stopping training: {stop_reason}")

            acc_loss.zero_()

            # Broadcast stop signal to all ranks to avoid race condition
            should_stop_tensor = torch.tensor(should_stop, device=device, dtype=torch.bool)
            dist.broadcast(should_stop_tensor, src=0)
            if should_stop_tensor.item():
                break
            if t_step_start is not None:
                t_step_start = time.time()

    if args.save_checkpoint:
        # full_state_dict=True + cpu_offload=True gathers the full dict
        # onto rank 0 only; other ranks get {}. Unbank + save runs rank-0.
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_state_dict = get_model_state_dict(model, options=options)
        if rank == 0:
            if use_banks:
                full_state_dict = unbank_state_dict(full_state_dict, banks)
            checkpoint_dir = f"checkpoints/{run_name}_step{global_step}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir, state_dict=full_state_dict)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Checkpoint saved to {checkpoint_dir}")

    # cleanup
    dist.barrier()
    dist.destroy_process_group()
