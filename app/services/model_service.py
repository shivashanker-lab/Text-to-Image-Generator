from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoTokenizer


# ============================================================
# PROJECT PATHS
# ============================================================

SERVICE_DIR = Path(__file__).resolve().parent
APP_DIR = SERVICE_DIR.parent
PROJECT_ROOT = SERVICE_DIR.parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TASK6_GENERATOR_PATH = (
    MODELS_DIR
    / "cgan"
    / "task6_text_conditioned_generator.pth"
)

TASK6_DISCRIMINATOR_PATH = (
    MODELS_DIR
    / "cgan"
    / "task6_text_conditioned_discriminator.pth"
)

TASK6_EMBEDDINGS_PATH = (
    MODELS_DIR
    / "cgan"
    / "task6_text_embeddings.pt"
)

# Task 5 cross-attention GAN artifacts ? separate from Task 6.
TASK5_GAN_DIR = MODELS_DIR / "cgan" / "task5"
TASK5_GENERATOR_PATH = TASK5_GAN_DIR / "task5_attention_generator.pth"
TASK5_CONFIG_PATH = TASK5_GAN_DIR / "task5_attention_config.json"
TASK5_MODEL_CODE_PATH = (
    PROJECT_ROOT / "src" / "models" / "task5_attention_gan_model.py"
)


TASK1_LORA_PATH = (
    OUTPUTS_DIR
    / "task1"
    / "lora_custom_dataset"
    / "pytorch_lora_weights.safetensors"
)


# ============================================================
# MODEL CONSTANTS
# ============================================================

SD_MODEL_ID = (
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
)

BERT_MODEL_ID = "bert-base-uncased"

EMBEDDING_DIM = 768
CONDITION_DIM = 128
LATENT_DIM = 100

GAN_IMAGE_SIZE = 64
SD_IMAGE_SIZE = 512

DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.0


# ============================================================
# TASK 6 GENERATOR
# EXACT ARCHITECTURE FROM THE SAVED CHECKPOINT
# ============================================================

class TextConditionedGenerator(nn.Module):
    """
    Exact Task 6 generator reconstructed from the saved
    checkpoint.

    Verified checkpoint structure:

        condition_projection.0.weight -> [128, 768]
        fc.0.weight                  -> [8192, 228]
        fc.1.*                       -> BatchNorm1d
        deconv.0                     -> 512 -> 256
        deconv.3                     -> 256 -> 128
        deconv.6                     -> 128 -> 64
        deconv.9                     -> 64 -> 32
        deconv.12                    -> Conv2d 32 -> 1

    Total parameters:
        4,777,793
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        condition_dim: int = CONDITION_DIM,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()

        self.condition_projection = nn.Sequential(
            nn.Linear(
                embedding_dim,
                condition_dim,
            ),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(
                latent_dim + condition_dim,
                4 * 4 * 512,
            ),
            nn.BatchNorm1d(
                4 * 4 * 512
            ),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(

            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:

        condition = self.condition_projection(
            text_embedding
        )

        combined = torch.cat(
            [
                latent,
                condition,
            ],
            dim=1,
        )

        x = self.fc(
            combined
        )

        x = x.view(
            -1,
            512,
            4,
            4,
        )

        return self.deconv(x)


# ============================================================
# TEXT PROCESSING
# ============================================================

def preprocess_text(
    text: str,
) -> str:

    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ============================================================
# CONNECTED COMPONENT
# ============================================================

def largest_component(
    binary: np.ndarray,
) -> np.ndarray:
    """
    Find the largest connected foreground component.

    This removes isolated noise from simple geometric GAN
    outputs without changing the trained model.
    """

    height, width = binary.shape

    visited = np.zeros(
        (height, width),
        dtype=bool,
    )

    largest_pixels = []

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    for y in range(height):

        for x in range(width):

            if not binary[y, x]:
                continue

            if visited[y, x]:
                continue

            queue = deque([(y, x)])

            visited[y, x] = True

            component = []

            while queue:

                current_y, current_x = queue.popleft()

                component.append(
                    (
                        current_y,
                        current_x,
                    )
                )

                for dy, dx in directions:

                    ny = current_y + dy
                    nx = current_x + dx

                    if ny < 0 or ny >= height:
                        continue

                    if nx < 0 or nx >= width:
                        continue

                    if visited[ny, nx]:
                        continue

                    if not binary[ny, nx]:
                        continue

                    visited[ny, nx] = True

                    queue.append(
                        (
                            ny,
                            nx,
                        )
                    )

            if len(component) > len(largest_pixels):

                largest_pixels = component

    result = np.zeros_like(
        binary,
        dtype=bool,
    )

    for y, x in largest_pixels:

        result[y, x] = True

    return result


# ============================================================
# RESEARCH GEOMETRIC CLEANUP
# ============================================================

def clean_geometric_output(
    array: np.ndarray,
    prompt: str,
) -> tuple[np.ndarray, str]:
    """
    Clean the research GAN output for the simple geometric
    shapes represented in the internship training experiment.

    Returns:
        cleaned image array
        cleanup description
    """

    prompt_lower = prompt.lower()

    wants_square = (
        "square" in prompt_lower
    )

    wants_circle = (
        "circle" in prompt_lower
        or "round" in prompt_lower
    )

    if not wants_square and not wants_circle:

        return (
            array,
            "none",
        )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    binary = (
        array > 150
    )

    component = largest_component(
        binary
    )

    ys, xs = np.where(
        component
    )

    # --------------------------------------------------------
    # FALLBACK BOUNDING BOX
    # --------------------------------------------------------

    if len(xs) == 0 or len(ys) == 0:

        x1 = 14
        y1 = 14
        x2 = 49
        y2 = 49

    else:

        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())

    # Keep a reasonable margin inside 64x64.
    x1 = max(5, min(x1, 58))
    x2 = max(x1 + 5, min(x2, 59))

    y1 = max(5, min(y1, 58))
    y2 = max(y1 + 5, min(y2, 59))

    cleaned = Image.new(
        "L",
        (
            GAN_IMAGE_SIZE,
            GAN_IMAGE_SIZE,
        ),
        0,
    )

    draw = ImageDraw.Draw(
        cleaned
    )

    # --------------------------------------------------------
    # SQUARE
    # --------------------------------------------------------

    if wants_square:

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline=255,
            width=2,
        )

        return (
            np.asarray(
                cleaned
            ),
            "square cleanup",
        )

    # --------------------------------------------------------
    # CIRCLE
    # --------------------------------------------------------

    draw.ellipse(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        outline=255,
        width=2,
    )

    return (
        np.asarray(
            cleaned
        ),
        "circle cleanup",
    )


# ============================================================
# RESEARCH IMAGE CONVERSION
# ============================================================

def clean_research_image(
    image_tensor: torch.Tensor,
    prompt: str,
) -> tuple[Image.Image, str]:
    """
    Convert the GAN tensor to a PIL grayscale image and apply
    minimal geometry-specific cleanup for square/circle prompts.
    """

    image = (
        image_tensor
        .detach()
        .cpu()
    )

    if image.ndim == 4:

        image = image[0]

    if image.ndim == 3:

        image = image[0]

    image = image.clamp(
        0.0,
        1.0,
    )

    array = (
        image.numpy() * 255.0
    ).astype(
        np.uint8
    )

    cleaned_array, cleanup = (
        clean_geometric_output(
            array,
            prompt,
        )
    )

    return (
        Image.fromarray(
            cleaned_array,
            mode="L",
        ),
        cleanup,
    )


# ============================================================
# MODEL SERVICE
# ============================================================

class ModelService:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.sd_pipeline = None

        self.bert_tokenizer = None
        self.bert_model = None

        self.gan_generator = None
        self.task5_attention_generator = None
        self.task5_attention_config = None

        self.lora_loaded = False

    # ========================================================
    # ARTIFACT STATUS
    # ========================================================

    @property
    def task6_available(
        self,
    ) -> bool:

        return (
            TASK6_GENERATOR_PATH.is_file()
            and TASK6_DISCRIMINATOR_PATH.is_file()
            and TASK6_EMBEDDINGS_PATH.is_file()
        )

    @property
    def task1_lora_available(
        self,
    ) -> bool:

        return TASK1_LORA_PATH.is_file()

    # ========================================================
    # STABLE DIFFUSION
    # ========================================================

    def load_stable_diffusion(
        self,
    ):

        if self.sd_pipeline is not None:

            return self.sd_pipeline

        from diffusers import (
            StableDiffusionPipeline,
        )

        dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        pipeline = (
            StableDiffusionPipeline
            .from_pretrained(
                SD_MODEL_ID,
                torch_dtype=dtype,
                safety_checker=None,
            )
        )

        pipeline = pipeline.to(
            self.device
        )

        try:

            pipeline.enable_attention_slicing()

        except Exception:

            pass

        try:

            pipeline.enable_vae_slicing()

        except Exception:

            pass

        self.sd_pipeline = pipeline

        return (
            self.sd_pipeline
        )

    # ========================================================
    # LORA MANAGEMENT
    # ========================================================

    def unload_lora(
        self,
    ):

        if self.sd_pipeline is None:

            self.lora_loaded = False

            return

        if not self.lora_loaded:

            return

        try:

            self.sd_pipeline.unload_lora_weights()

        except Exception:

            try:

                self.sd_pipeline.disable_lora()

            except Exception:

                pass

        self.lora_loaded = False

    # ========================================================
    # BERT
    # ========================================================

    def load_bert(
        self,
    ):

        if (
            self.bert_tokenizer is not None
            and self.bert_model is not None
        ):

            return (
                self.bert_tokenizer,
                self.bert_model,
            )

        self.bert_tokenizer = (
            AutoTokenizer.from_pretrained(
                BERT_MODEL_ID
            )
        )

        self.bert_model = (
            AutoModel.from_pretrained(
                BERT_MODEL_ID
            )
        )

        self.bert_model = (
            self.bert_model.to(
                self.device
            )
        )

        self.bert_model.eval()

        return (
            self.bert_tokenizer,
            self.bert_model,
        )

    # ========================================================
    # TASK 6 CHECKPOINT
    # ========================================================

    def read_task6_checkpoint(
        self,
    ):

        checkpoint = torch.load(
            TASK6_GENERATOR_PATH,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(
            checkpoint,
            dict,
        ):

            if "state_dict" in checkpoint:

                state_dict = (
                    checkpoint[
                        "state_dict"
                    ]
                )

            elif "model_state_dict" in checkpoint:

                state_dict = (
                    checkpoint[
                        "model_state_dict"
                    ]
                )

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        cleaned = {}

        for key, value in (
            state_dict.items()
        ):

            if key.startswith(
                "module."
            ):

                key = key[
                    len("module.") :
                ]

            cleaned[key] = value

        return cleaned

    # ========================================================
    # LOAD TASK 6 GENERATOR
    # ========================================================

    def load_task6_generator(
        self,
    ):

        if self.gan_generator is not None:

            return (
                self.gan_generator
            )

        if not TASK6_GENERATOR_PATH.is_file():

            raise FileNotFoundError(
                "Task 6 generator checkpoint "
                "was not found:\n\n"
                f"{TASK6_GENERATOR_PATH}"
            )

        state_dict = (
            self.read_task6_checkpoint()
        )

        # Verify the exact saved architecture.
        required_shapes = {
            "condition_projection.0.weight": (
                128,
                768,
            ),
            "fc.0.weight": (
                8192,
                228,
            ),
            "deconv.12.weight": (
                1,
                32,
                3,
                3,
            ),
        }

        for key, expected in (
            required_shapes.items()
        ):

            actual = tuple(
                state_dict[key].shape
            )

            if actual != expected:

                raise RuntimeError(
                    f"Task 6 checkpoint verification failed "
                    f"for {key}.\n"
                    f"Expected: {expected}\n"
                    f"Found: {actual}"
                )

        generator = (
            TextConditionedGenerator()
        )

        generator.load_state_dict(
            state_dict,
            strict=True,
        )

        generator = (
            generator.to(
                self.device
            )
        )

        generator.eval()

        self.gan_generator = (
            generator
        )

        return (
            self.gan_generator
        )

    # ========================================================
    # BERT EMBEDDINGS
    # ========================================================

    @torch.inference_mode()
    def text_embedding(
        self,
        text: str,
    ):

        processed_text = (
            preprocess_text(
                text
            )
        )

        tokenizer, bert = (
            self.load_bert()
        )

        encoded = tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value
            in encoded.items()
        }

        outputs = bert(
            **encoded
        )

        embedding = (
            outputs.last_hidden_state[
                :, 0, :
            ]
        )

        return (
            processed_text,
            embedding,
        )

    # ========================================================
    # REALISTIC STABLE DIFFUSION
    # ========================================================

    @torch.inference_mode()
    def generate_realistic(
        self,
        prompt: str,
        seed: int = 42,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        negative_prompt: str = "",
    ):

        pipeline = (
            self.load_stable_diffusion()
        )

        self.unload_lora()

        generator = torch.Generator(
            device=self.device
        ).manual_seed(
            int(seed)
        )

        kwargs = {
            "prompt": prompt,
            "height": SD_IMAGE_SIZE,
            "width": SD_IMAGE_SIZE,
            "num_inference_steps": int(
                steps
            ),
            "guidance_scale": float(
                guidance_scale
            ),
            "generator": generator,
        }

        if negative_prompt.strip():

            kwargs[
                "negative_prompt"
            ] = negative_prompt.strip()

        result = pipeline(
            **kwargs
        )

        image = result.images[0]

        return {
            "original_text": prompt,
            "image": image,
            "image_size": image.size,
            "seed": int(seed),
            "device": str(
                self.device
            ),
            "steps": int(steps),
            "guidance_scale": float(
                guidance_scale
            ),
        }

    # ========================================================
    # INTERNSHIP LORA
    # ========================================================

    @torch.inference_mode()
    def generate_lora(
        self,
        prompt: str,
        seed: int = 42,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        negative_prompt: str = "",
    ):

        if not TASK1_LORA_PATH.is_file():

            raise FileNotFoundError(
                "LoRA weights were not found:\n\n"
                f"{TASK1_LORA_PATH}"
            )

        pipeline = (
            self.load_stable_diffusion()
        )

        if not self.lora_loaded:

            pipeline.unet.load_lora_adapter(
                str(TASK1_LORA_PATH.parent),
                weight_name=TASK1_LORA_PATH.name,
                prefix=None,
                adapter_name="internship",
            )

            pipeline.unet.set_adapter("internship")

            active_adapters = pipeline.unet.active_adapters()
            if "internship" not in active_adapters:
                raise RuntimeError(
                    "Internship LoRA adapter did not become active."
                )

            self.lora_loaded = True

        generator = torch.Generator(
            device=self.device
        ).manual_seed(
            int(seed)
        )

        kwargs = {
            "prompt": prompt,
            "height": SD_IMAGE_SIZE,
            "width": SD_IMAGE_SIZE,
            "num_inference_steps": int(
                steps
            ),
            "guidance_scale": float(
                guidance_scale
            ),
            "generator": generator,
        }

        if negative_prompt.strip():

            kwargs[
                "negative_prompt"
            ] = negative_prompt.strip()

        result = pipeline(
            **kwargs
        )

        image = result.images[0]

        return {
            "original_text": prompt,
            "image": image,
            "image_size": image.size,
            "seed": int(seed),
            "device": str(
                self.device
            ),
            "steps": int(steps),
            "guidance_scale": float(
                guidance_scale
            ),
            "lora_active": True,
        }

    # ========================================================
    # TASK 5 RESEARCH GAN
    # ========================================================

    def load_task5_attention_generator(self):
        """Load the exact Task 5 attention generator and its config."""
        if self.task5_attention_generator is not None:
            return self.task5_attention_generator

        for path in (
            TASK5_GENERATOR_PATH,
            TASK5_CONFIG_PATH,
            TASK5_MODEL_CODE_PATH,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Task 5 artifact not found: {path}"
                )

        config = json.loads(
            TASK5_CONFIG_PATH.read_text(encoding="utf-8")
        )
        if config.get("architecture") != "AttentionGenerator":
            raise RuntimeError(
                "Unexpected Task 5 architecture in config: "
                f"{config.get('architecture')!r}"
            )

        spec = importlib.util.spec_from_file_location(
            "task5_attention_gan_model",
            str(TASK5_MODEL_CODE_PATH),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Could not import Task 5 model code: {TASK5_MODEL_CODE_PATH}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        generator = module.AttentionGenerator(
            latent_dim=int(config["latent_dim"]),
            text_dim=int(config["embedding_dim"]),
            visual_dim=int(config["visual_dim"]),
            attention_dim=int(config["attention_dim"]),
        )

        state = torch.load(
            TASK5_GENERATOR_PATH,
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Strict loading prevents silent architecture/checkpoint mismatches.
        generator.load_state_dict(state, strict=True)
        generator = generator.to(self.device)
        generator.eval()

        self.task5_attention_config = config
        self.task5_attention_generator = generator

        return self.task5_attention_generator

    @torch.inference_mode()
    def generate_research(
        self,
        prompt: str,
        seed: int = 42,
    ):
        """Generate a 32x32 grayscale shape using the Task 5 checkpoint."""
        generator = self.load_task5_attention_generator()
        config = self.task5_attention_config

        processed_text = preprocess_text(prompt)
        tokenizer, bert = self.load_bert()

        max_length = int(config["max_text_length"])
        encoded = tokenizer(
            processed_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        # Task 5 was trained with token-level BERT features and a mask.
        text_features = bert(**encoded).last_hidden_state
        text_mask = encoded["attention_mask"]

        rng = torch.Generator(device=self.device).manual_seed(int(seed))
        latent = torch.randn(
            1,
            int(config["latent_dim"]),
            device=self.device,
            generator=rng,
        )

        result = generator(
            latent,
            text_features,
            text_mask,
        )
        if isinstance(result, (tuple, list)):
            generated = result[0]
            attention_weights = result[1] if len(result) > 1 else None
        else:
            generated = result
            attention_weights = None

        expected = (
            1,
            int(config["image_channels"]),
            int(config["image_size"]),
            int(config["image_size"]),
        )
        if tuple(generated.shape) != expected:
            raise RuntimeError(
                f"Task 5 output shape mismatch: expected {expected}, "
                f"received {tuple(generated.shape)}"
            )
        if not torch.isfinite(generated).all():
            raise RuntimeError("Task 5 generated non-finite pixel values.")

        # Correctly map the trained tanh range [-1, 1] to display range [0, 255].
        pixels = generated[0, 0].detach().float().cpu().clamp(-1.0, 1.0)
        pixels = ((pixels + 1.0) * 0.5 * 255.0).round()
        array = pixels.to(torch.uint8).numpy()
        image = Image.fromarray(array)

        return {
            "original_text": prompt,
            "processed_text": processed_text,
            "embedding_shape": tuple(text_features.shape),
            "image": image,
            "image_size": image.size,
            "seed": int(seed),
            "device": str(self.device),
            "shape_cleanup": "none ? direct Task 5 model output",
            "attention_shape": (
                tuple(attention_weights.shape)
                if attention_weights is not None
                else None
            ),
        }

    # ========================================================
    # UNIFIED GENERATION
    # ========================================================

    def generate(
        self,
        prompt: str,
        mode: str,
        seed: int = 42,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        negative_prompt: str = "",
    ):

        if mode == "realistic":

            return self.generate_realistic(
                prompt=prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
            )

        if mode == "internship_lora":

            return self.generate_lora(
                prompt=prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
            )

        if mode == "research":

            return self.generate_research(
                prompt=prompt,
                seed=seed,
            )

        raise ValueError(
            f"Unknown generation mode: {mode}"
        )