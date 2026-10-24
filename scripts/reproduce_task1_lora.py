from pathlib import Path
import gc

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1"
LORA_OUTPUT_DIR = OUTPUT_ROOT / "lora_custom_dataset"
BASELINE_DIR = OUTPUT_ROOT / "baseline"
ADAPTED_DIR = OUTPUT_ROOT / "adapted"
TRAINING_DIR = OUTPUT_ROOT / "training"
COMPARISON_DIR = OUTPUT_ROOT / "comparison"

for directory in [
    OUTPUT_ROOT,
    LORA_OUTPUT_DIR,
    BASELINE_DIR,
    ADAPTED_DIR,
    TRAINING_DIR,
    COMPARISON_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASET_ID = "intro/flickr8k"

NUM_SAMPLES = 60
IMAGE_SIZE = 256
BATCH_SIZE = 1
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_TEXT_LENGTH = 77

LORA_RANK = 4
LORA_ALPHA = 4
LORA_DROPOUT = 0.0
ADAPTER_NAME = "custom_dataset_adapter"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TASK 1 LORA CHECKPOINT REPRODUCTION - STABLE VERSION")
print("=" * 70)

print("Project root:", PROJECT_ROOT)
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Model:", MODEL_ID)
print("Dataset:", DATASET_ID)
print("Training images:", NUM_SAMPLES)
print("Epochs:", NUM_EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)

print("=" * 70)


# ============================================================
# LOAD FLICKR8K
# ============================================================

print("\n===== LOADING FLICKR8K =====")

dataset = load_dataset(
    DATASET_ID,
    split="train",
    streaming=True,
)

training_images = []
training_captions = []

dataset_stream = iter(dataset)

for index in range(NUM_SAMPLES):

    sample = next(dataset_stream)

    image = sample["image"].convert("RGB")
    caption = sample["caption_0"].strip()

    training_images.append(image)
    training_captions.append(caption)

    if (index + 1) % 10 == 0:
        print(f"Loaded {index + 1}/{NUM_SAMPLES}")

print("\nDataset preparation complete.")
print("Images:", len(training_images))
print("Captions:", len(training_captions))

assert len(training_images) == NUM_SAMPLES
assert len(training_captions) == NUM_SAMPLES

print("\nFirst three captions:")

for i, caption in enumerate(training_captions[:3], start=1):
    print(f"{i}. {caption}")


# ============================================================
# LOAD STABLE DIFFUSION
# ============================================================

print("\n===== LOADING STABLE DIFFUSION =====")

pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    local_files_only=True,
).to(device)

print("Stable Diffusion loaded.")


# ============================================================
# GENERATE BASELINE IMAGES
# ============================================================

print("\n===== GENERATING BASELINE IMAGES =====")

baseline_images = []
baseline_prompts = training_captions[:3]

pipeline.unet.eval()

for i, prompt in enumerate(baseline_prompts, start=1):

    print(f"Generating baseline image {i}/3...")
    print("Prompt:", prompt)

    generator = torch.Generator(
        device=device
    ).manual_seed(40 + i)

    with torch.inference_mode():

        result = pipeline(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
        )

    image = result.images[0]

    baseline_images.append(image)

    path = BASELINE_DIR / f"baseline_{i}.png"
    image.save(path)

    print("Saved:", path)

    del result

    if device == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# TRAINING DATASET
# ============================================================

print("\n===== PREPARING TRAINING DATASET =====")

image_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ),
])


class ImageCaptionDataset(Dataset):

    def __init__(self, images, captions):
        self.images = images
        self.captions = captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = image_transform(
            self.images[index]
        )

        caption = self.captions[index]

        tokens = pipeline.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
        }


training_dataset = ImageCaptionDataset(
    training_images,
    training_captions,
)

training_loader = DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

print("Dataset size:", len(training_dataset))
print("Batches per epoch:", len(training_loader))

assert len(training_dataset) == 60
assert len(training_loader) == 60


# ============================================================
# CONFIGURE LORA
# ============================================================

print("\n===== CONFIGURING LORA =====")

pipeline.unet.requires_grad_(False)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
)

pipeline.unet.add_adapter(
    lora_config,
    adapter_name=ADAPTER_NAME,
)

pipeline.unet.set_adapter(
    ADAPTER_NAME
)


# ============================================================
# IMPORTANT NUMERICAL-STABILITY FIX
# ============================================================
#
# Base UNet remains FP16 for GPU memory efficiency.
# Trainable LoRA parameters are explicitly FP32.
#
# This is the key change that prevents the FP16 LoRA
# parameters from becoming numerically unstable.
# ============================================================

for parameter in pipeline.unet.parameters():

    if parameter.requires_grad:
        parameter.data = parameter.data.float()


# ============================================================
# VAE / TEXT ENCODER
# ============================================================

pipeline.vae.to(
    device=device,
    dtype=torch.float32,
)

pipeline.vae.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)

pipeline.vae.eval()
pipeline.text_encoder.eval()
pipeline.unet.train()


# ============================================================
# VERIFY PARAMETERS
# ============================================================

trainable_parameters = [
    parameter
    for parameter in pipeline.unet.parameters()
    if parameter.requires_grad
]

total_unet_parameters = sum(
    parameter.numel()
    for parameter in pipeline.unet.parameters()
)

trainable_parameter_count = sum(
    parameter.numel()
    for parameter in pipeline.unet.parameters()
    if parameter.requires_grad
)

trainable_ratio = (
    100.0
    * trainable_parameter_count
    / total_unet_parameters
)

print("Total UNet parameters:", total_unet_parameters)
print("Trainable LoRA parameters:", trainable_parameter_count)
print(f"Trainable ratio: {trainable_ratio:.4f}%")

print(
    "Trainable parameter dtype:",
    trainable_parameters[0].dtype,
)

print(
    "Active adapters:",
    pipeline.unet.active_adapters(),
)

assert trainable_parameter_count == 797184
assert trainable_parameters[0].dtype == torch.float32
assert ADAPTER_NAME in pipeline.unet.active_adapters()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE,
)

pipeline.unet.enable_gradient_checkpointing()

print("Optimizer:", type(optimizer).__name__)
print("Gradient checkpointing: enabled")


# ============================================================
# AMP GRADIENT SCALER
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device == "cuda"),
)

print(
    "Gradient scaler enabled:",
    scaler.is_enabled(),
)


# ============================================================
# NUMERICAL STABILITY CHECK
# ============================================================

print("\n===== NUMERICAL STABILITY CHECK =====")

test_batch = next(iter(training_loader))

test_pixels = test_batch["pixel_values"].to(
    device=device,
    dtype=torch.float32,
)

with torch.no_grad():

    test_latents = pipeline.vae.encode(
        test_pixels
    ).latent_dist.sample()

    test_latents = (
        test_latents
        * pipeline.vae.config.scaling_factor
    )

print(
    "Latent shape:",
    tuple(test_latents.shape),
)

print(
    "Latents contain NaN:",
    torch.isnan(test_latents).any().item(),
)

print(
    "Latents contain Inf:",
    torch.isinf(test_latents).any().item(),
)

assert torch.isfinite(test_latents).all()

del test_batch
del test_pixels
del test_latents

if device == "cuda":
    torch.cuda.empty_cache()


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING STABLE LORA FINE-TUNING")
print("=" * 70)

noise_scheduler = pipeline.scheduler

training_losses = []
global_step = 0

pipeline.unet.train()

for epoch in range(NUM_EPOCHS):

    print(
        f"\n===== EPOCH {epoch + 1}/{NUM_EPOCHS} ====="
    )

    epoch_losses = []

    for batch in training_loader:

        pixel_values = batch["pixel_values"].to(
            device=device,
            dtype=torch.float32,
        )

        input_ids = batch["input_ids"].to(device)

        # ----------------------------------------------------
        # IMAGE + TEXT ENCODING
        # ----------------------------------------------------

        with torch.no_grad():

            latents = pipeline.vae.encode(
                pixel_values
            ).latent_dist.sample()

            latents = (
                latents
                * pipeline.vae.config.scaling_factor
            )

            encoder_hidden_states = pipeline.text_encoder(
                input_ids
            )[0]

        # ----------------------------------------------------
        # NOISE
        # ----------------------------------------------------

        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
        ).long()

        noisy_latents = noise_scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device == "cuda"),
        ):

            model_output = pipeline.unet(
                noisy_latents.to(dtype),
                timesteps,
                encoder_hidden_states.to(dtype),
            )

            model_pred = model_output.sample

        # Check forward output before calculating loss.
        if not torch.isfinite(model_pred).all():

            raise RuntimeError(
                f"Non-finite UNet output at step "
                f"{global_step + 1}."
            )

        # ----------------------------------------------------
        # DIFFUSION TARGET
        # ----------------------------------------------------

        if (
            noise_scheduler.config.prediction_type
            == "epsilon"
        ):

            target = noise

        elif (
            noise_scheduler.config.prediction_type
            == "v_prediction"
        ):

            target = noise_scheduler.get_velocity(
                latents,
                noise,
                timesteps,
            )

        else:

            raise ValueError(
                "Unsupported prediction type: "
                f"{noise_scheduler.config.prediction_type}"
            )

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = F.mse_loss(
            model_pred.float(),
            target.float(),
            reduction="mean",
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss at step "
                f"{global_step + 1}: "
                f"{loss.item()}"
            )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        # ----------------------------------------------------
        # RECORD LOSS
        # ----------------------------------------------------

        loss_value = (
            loss.detach()
            .float()
            .item()
        )

        training_losses.append(loss_value)
        epoch_losses.append(loss_value)

        global_step += 1

        if global_step % 10 == 0:

            print(
                f"Step {global_step:03d} | "
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Loss: {loss_value:.6f}"
            )

        del (
            pixel_values,
            input_ids,
            latents,
            encoder_hidden_states,
            noise,
            timesteps,
            noisy_latents,
            model_output,
            model_pred,
            target,
            loss,
        )

    average_epoch_loss = (
        sum(epoch_losses)
        / len(epoch_losses)
    )

    print(
        f"Epoch {epoch + 1} completed | "
        f"Average Loss: {average_epoch_loss:.6f}"
    )

    if device == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# TRAINING VERIFICATION
# ============================================================

print("\n===== TRAINING VERIFICATION =====")

print("Total training steps:", global_step)
print("Recorded loss values:", len(training_losses))

assert global_step == 180
assert len(training_losses) == 180
assert all(
    torch.isfinite(torch.tensor(value))
    for value in training_losses
)

print("PASS - 180 training steps completed.")
print("PASS - All loss values are finite.")


# ============================================================
# SAVE TRAINING LOSS
# ============================================================

loss_plot_path = (
    TRAINING_DIR
    / "training_loss.png"
)

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(training_losses) + 1),
    training_losses,
    linewidth=1,
)

plt.xlabel("Training Step")
plt.ylabel("Diffusion Training Loss")
plt.title(
    "Stable Diffusion LoRA Fine-Tuning Loss"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    loss_plot_path,
    dpi=150,
)

plt.close()

print(
    "Loss plot saved:",
    loss_plot_path,
)


# ============================================================
# SAVE LORA
# ============================================================

print("\n===== SAVING LORA WEIGHTS =====")

lora_weight_path = (
    LORA_OUTPUT_DIR
    / "pytorch_lora_weights.safetensors"
)

pipeline.unet.save_lora_adapter(
    str(LORA_OUTPUT_DIR),
    adapter_name=ADAPTER_NAME,
    weight_name="pytorch_lora_weights.safetensors",
    safe_serialization=True,
)

assert lora_weight_path.exists()
assert lora_weight_path.stat().st_size > 0

file_size_mb = (
    lora_weight_path.stat().st_size
    / (1024 ** 2)
)

print(
    "LoRA file:",
    lora_weight_path,
)

print(
    f"LoRA size: {file_size_mb:.2f} MB"
)


# ============================================================
# FREE TRAINING MEMORY
# ============================================================

del optimizer
del scaler
del training_loader
del training_dataset
del pipeline

gc.collect()

if device == "cuda":
    torch.cuda.empty_cache()


# ============================================================
# RELOAD FRESH PIPELINE
# ============================================================

print("\n===== RELOADING FRESH PIPELINE =====")

reload_pipeline = (
    StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        local_files_only=True,
    ).to(device)
)

reload_pipeline.unet.requires_grad_(False)

reload_lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
)

reload_pipeline.unet.add_adapter(
    reload_lora_config,
    adapter_name=ADAPTER_NAME,
)

saved_lora_state = load_file(
    str(lora_weight_path),
    device="cpu",
)

print(
    "Saved LoRA tensors:",
    len(saved_lora_state),
)

set_peft_model_state_dict(
    reload_pipeline.unet,
    saved_lora_state,
    adapter_name=ADAPTER_NAME,
)

reload_pipeline.unet.set_adapter(
    ADAPTER_NAME
)

loaded_trainable_parameters = sum(
    parameter.numel()
    for parameter in reload_pipeline.unet.parameters()
    if parameter.requires_grad
)

print(
    "Reloaded trainable parameters:",
    loaded_trainable_parameters,
)

print(
    "Active adapters:",
    reload_pipeline.unet.active_adapters(),
)

assert len(saved_lora_state) > 0
assert loaded_trainable_parameters == 797184
assert (
    ADAPTER_NAME
    in reload_pipeline.unet.active_adapters()
)

print(
    "PASS - LoRA weights reloaded successfully."
)


# ============================================================
# GENERATE ADAPTED IMAGES
# ============================================================

print("\n===== GENERATING ADAPTED IMAGES =====")

reload_pipeline.unet.eval()

adapted_images = []

for i, prompt in enumerate(
    baseline_prompts,
    start=1,
):

    print(
        f"\nGenerating adapted image {i}/3"
    )

    print(
        "Prompt:",
        prompt,
    )

    generator = torch.Generator(
        device=device
    ).manual_seed(42 + i)

    with torch.inference_mode():

        result = reload_pipeline(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
        )

    image = result.images[0]

    adapted_images.append(image)

    image_path = (
        ADAPTED_DIR
        / f"adapted_{i}.png"
    )

    image.save(image_path)

    print(
        "Saved:",
        image_path,
    )

    del result

    if device == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# VERIFY OUTPUTS
# ============================================================

print(
    "\n===== ADAPTED IMAGE VERIFICATION ====="
)

for i in range(1, 4):

    image_path = (
        ADAPTED_DIR
        / f"adapted_{i}.png"
    )

    assert image_path.exists()

    print(
        "PASS -",
        image_path,
    )


# ============================================================
# BASELINE VS ADAPTED COMPARISON
# ============================================================

print(
    "\n===== CREATING COMPARISON ====="
)

comparison_path = (
    COMPARISON_DIR
    / "baseline_vs_adapted_comparison.png"
)

figure, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10),
)

for i in range(3):

    axes[0, i].imshow(
        baseline_images[i]
    )

    axes[0, i].set_title(
        f"Baseline {i + 1}"
    )

    axes[0, i].axis("off")

    axes[1, i].imshow(
        adapted_images[i]
    )

    axes[1, i].set_title(
        f"LoRA Adapted {i + 1}"
    )

    axes[1, i].axis("off")

plt.tight_layout()

figure.savefig(
    comparison_path,
    dpi=150,
)

plt.close(figure)

print(
    "Comparison saved:",
    comparison_path,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TASK 1 LORA REPRODUCTION COMPLETE")
print("=" * 70)

print("Training examples:", NUM_SAMPLES)
print("Epochs:", NUM_EPOCHS)
print("Training steps:", global_step)
print("LoRA parameters:", trainable_parameter_count)
print(
    f"LoRA ratio: {trainable_ratio:.4f}%"
)

print(
    f"LoRA file size: {file_size_mb:.2f} MB"
)

print("\nArtifacts:")

print(
    "LoRA:",
    lora_weight_path,
)

print(
    "Loss plot:",
    loss_plot_path,
)

print(
    "Comparison:",
    comparison_path,
)

for i in range(1, 4):

    print(
        f"Baseline {i}:",
        BASELINE_DIR
        / f"baseline_{i}.png",
    )

    print(
        f"Adapted {i}:",
        ADAPTED_DIR
        / f"adapted_{i}.png",
    )

print("\nPASS - TASK 1 LORA PIPELINE REPRODUCED")
print("=" * 70)