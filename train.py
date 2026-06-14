import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
import argparse
import json
from datetime import datetime

from himuon.dataset import ToyDataset
from himuon.model import get_model_and_tokenizer, count_model_parameters, count_num_training_tokens
from himuon.optim import get_optimizer
from himuon.utils import TimeBudget, seed_everything, merge_kv_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument("--reconfigure-schedule", type=str, default="", help='JSON list of dicts; each must have "step" key, other keys forwarded to optimizer.reconfigure(). Applied at matching global_step (step=0 fires before training). Example: \'[{"step":0,"tile_size":1000000000},{"step":100,"tile_size":512}]\'')
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True, help="CUDA-graph capture for HiMuon (default on). Use --no-cuda-graph to disable. Controls zero_grad(set_to_none=False) for stable grad pointers.")
    parser.add_argument("--time-budget", type=float, default=None, help="Stop training after X hours if provided")
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
    
    # setup logger
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = args.run_name or f"{args.model}_{args.optimizer}_lr{args.lr}_{timestamp}"
    if args.wandb:
        from himuon.logger import WandbLogger
        logger = WandbLogger(args, run_name)
    else:
        from himuon.logger import LoguruLogger
        logger = LoguruLogger(run_name)
    logger.info(args)
    
    # load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(args.model)
    model.to(device)
    logger.info(count_model_parameters(model)[1])

    # load streaming dataset, shuffle with buffer
    dataset = load_dataset("HuggingFaceFW/fineweb", name=args.dataset, split="train", streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=10000)
    # convert to on-the-fly tokenized dataloader
    toy_dataset = ToyDataset(dataset, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(toy_dataset, batch_size=args.batch_size, num_workers=0)

    # load eval dataset (different FineWeb subset for true held-out validation)
    # Pre-collect fixed batches so every eval pass uses identical data
    if args.eval_interval > 0:
        eval_dataset = load_dataset("HuggingFaceFW/fineweb", name=args.eval_dataset, split="train", streaming=True)
        eval_dataset = eval_dataset.shuffle(seed=args.seed, buffer_size=10000)
        eval_toy_dataset = ToyDataset(eval_dataset, tokenizer, max_length=args.max_length)
        _eval_iter = iter(DataLoader(eval_toy_dataset, batch_size=args.batch_size, num_workers=0))
        eval_batches = [next(_eval_iter) for _ in range(args.eval_steps)]
        del _eval_iter

    # load optimizer and scheduler
    if args.num_training_steps is None:
        num_training_tokens = count_num_training_tokens(model)
        tokens_per_step = args.batch_size * args.max_length * args.gradient_accumulation_steps
        args.num_training_steps = int(num_training_tokens / tokens_per_step)
    logger.info(f"Number of training steps: {args.num_training_steps}")
    optimizer = get_optimizer(args.optimizer, model, args)
    logger.log_optimizer_source(optimizer)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.num_training_steps)

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
            diff = ", ".join(f"{k} {old[k]} -> {new[k]}" for k in kwargs if old[k] != new[k])
            logger.info(f"step {at_step}, reconfigure: {diff or 'no change'}")

    apply_reconfigure(0)

    # train
    time_budget = TimeBudget(args.time_budget)
    if args.time_budget:
        logger.info(f"Training time budget remaining: {time_budget}")
    model.train()
    global_step = 0
    n_tokens = torch.tensor(0, dtype=torch.long, device=device)
    acc_loss = torch.tensor(0.0, device=device)
    for step, batch in enumerate(dataloader):
        n_tokens += (batch['labels'] != tokenizer.eos_token_id).sum()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        acc_loss += loss.detach()
        if (step+1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=not args.cuda_graph)
            global_step += 1
            apply_reconfigure(global_step)

            # Held-out eval
            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                model.eval()
                eval_loss = torch.tensor(0.0, device=device)
                with torch.no_grad():
                    for eval_batch in eval_batches:
                        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                        outputs = model(**eval_batch)
                        eval_loss += outputs.loss.detach()
                eval_loss_val = (eval_loss / args.eval_steps).item()
                model.train()
            else:
                eval_loss_val = None

            metrics = {
                "step": global_step,
                "lr": optimizer.param_groups[0]['lr'],
                "loss": acc_loss.item(),
                "tokens": n_tokens.item(),
            }
            if eval_loss_val is not None:
                metrics["eval_loss"] = eval_loss_val
            logger.log_metrics(metrics)
            acc_loss.zero_()
            if global_step >= args.num_training_steps:
                logger.info("Stopping training: Training finished")
                break
            if time_budget.is_expired():
                logger.info("Stopping training: Time budget exceeded.")
                break

    if args.save_checkpoint:
        checkpoint_dir = f"checkpoints/{run_name}_step{global_step}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved to {checkpoint_dir}")