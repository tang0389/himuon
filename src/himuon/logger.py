import os

try:
    import wandb
except ImportError:
    wandb = None


class LoguruLogger:
    def __init__(self, run_name: str):
        from loguru import logger

        self.logger = logger
        if run_name is not None and not os.path.exists(f"logs/{run_name}.log"):
            self.add(f"logs/{run_name}.log")

    def add(self, *args, **kwargs):
        self.logger.add(*args, **kwargs)

    def info(self, msg):
        self.logger.info(msg)

    def log_metrics(self, metrics: dict, verbose: bool = True):
        msg = f"Step: {metrics['step']} LR: {metrics['lr']:.6f} Training loss: {metrics['loss']:.4f}, Tokens: {metrics['tokens']:,}"
        if "eval_loss" in metrics:
            msg += f", Eval loss: {metrics['eval_loss']:.4f}"
        self.logger.info(msg)

    def log_optimizer_source(self, optimizer):
        pass


class WandbLogger:
    def __init__(self, config, run_name: str):
        if wandb is None:
            raise ImportError("wandb is not installed.")
        wandb.init(project="himuon", config=config, name=run_name)

    def add(self, *args, **kwargs):
        # WandbLogger does not maintain local logs
        pass

    def info(self, msg):
        # Use wandb.termlog for console output or just print
        wandb.termlog(str(msg))

    def log_metrics(self, metrics: dict, verbose: bool = True):
        wandb.log(metrics)
        if verbose:
            msg = f"Step: {metrics['step']} LR: {metrics['lr']:.6f} Training loss: {metrics['loss']:.4f}, Tokens: {metrics['tokens']:,}"
            wandb.termlog(str(msg))

    def log_optimizer_source(self, optimizer):
        import inspect

        src_file = inspect.getfile(type(optimizer))
        wandb.save(src_file, policy="now")
