from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


def get_model_and_tokenizer(
    model_name: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_name.lower().startswith("llama"):
        config = AutoConfig.from_pretrained(f"meta-llama/{model_name}")
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(
            f"meta-llama/{model_name}", trust_remote_code=True, use_fast=False
        )
        return model, tokenizer
    elif model_name.lower().startswith("qwen"):
        config = AutoConfig.from_pretrained(f"Qwen/{model_name}")
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(
            f"Qwen/{model_name}", trust_remote_code=True, use_fast=False
        )
        return model, tokenizer
    else:
        raise ValueError(f"Model {model_name} not supported")


def count_model_parameters(model: AutoModelForCausalLM) -> int:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n < 1e5:
        return n, f"{n:.0f} trainable parameters"
    elif n < 1e8:
        return n, f"{n / 1e6:.2f}M trainable parameters"
    else:
        return n, f"{n / 1e9:.2f}B trainable parameters"


def count_num_training_tokens(
    model: AutoModelForCausalLM, chinchilla_factor: float = 1.0
) -> int:
    """
    Note:
        Chinchilla scaling factor formula: num_tokens = 20 * chinchilla_factor * num_params
        Reference: https://arxiv.org/pdf/2203.15556
    """
    n, _ = count_model_parameters(model)
    return int(20 * chinchilla_factor * n)
