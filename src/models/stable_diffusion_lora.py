import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig


DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_stable_diffusion(
    model_id=DEFAULT_MODEL_ID,
    device=None,
):
    """
    Load a Stable Diffusion pipeline.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier.
    device : str or None
        Device to use. Defaults to CUDA when available.

    Returns
    -------
    StableDiffusionPipeline
        Loaded Stable Diffusion pipeline.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    return pipeline


def add_lora_adapter(
    pipeline,
    adapter_name="custom_dataset_adapter",
    rank=4,
    alpha=4,
    dropout=0.0,
):
    """
    Attach a LoRA adapter to the Stable Diffusion UNet.

    Parameters
    ----------
    pipeline : StableDiffusionPipeline
        Loaded Stable Diffusion pipeline.
    adapter_name : str
        Name of the LoRA adapter.
    rank : int
        LoRA rank.
    alpha : int
        LoRA scaling parameter.
    dropout : float
        LoRA dropout probability.

    Returns
    -------
    StableDiffusionPipeline
        Pipeline with the active LoRA adapter attached.
    """
    pipeline.unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
        lora_dropout=dropout,
        bias="none",
    )

    pipeline.unet.add_adapter(
        lora_config,
        adapter_name=adapter_name,
    )

    pipeline.unet.set_adapter(adapter_name)

    return pipeline
