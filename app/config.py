from __future__ import annotations

from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = APP_DIR.parent

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# TASK 1 — STABLE DIFFUSION + LORA
# ============================================================

TASK1_SD_MODEL_ID = (
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
)

TASK1_LORA_PATH = (
    OUTPUTS_DIR
    / "task1"
    / "lora_custom_dataset"
    / "pytorch_lora_weights.safetensors"
)

TASK1_IMAGE_SIZE = 512

TASK1_DEFAULT_STEPS = 30

TASK1_MIN_STEPS = 20

TASK1_MAX_STEPS = 40

TASK1_DEFAULT_GUIDANCE = 7.0

TASK1_MIN_GUIDANCE = 5.0

TASK1_MAX_GUIDANCE = 10.0


# ============================================================
# TASK 3 — BERT
# ============================================================

BERT_MODEL_ID = "bert-base-uncased"

EMBEDDING_DIM = 768


# ============================================================
# TASK 6 — CONDITIONAL GAN
# ============================================================

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

TASK6_IMAGE_SIZE = 64

TASK6_CONDITION_DIM = 128

TASK6_LATENT_DIM = 100


# ============================================================
# ATTENTION EXPERIMENTS
# ============================================================

COMPARISON_OUTPUT_DIR = (
    OUTPUTS_DIR / "comparisons"
)

ATTENTION_BASELINE_GENERATOR_LOSS = (
    COMPARISON_OUTPUT_DIR
    / "baseline_generator_losses.npy"
)

ATTENTION_GENERATOR_LOSS = (
    COMPARISON_OUTPUT_DIR
    / "attention_generator_losses.npy"
)

ATTENTION_BASELINE_DISCRIMINATOR_LOSS = (
    COMPARISON_OUTPUT_DIR
    / "baseline_discriminator_losses.npy"
)

ATTENTION_DISCRIMINATOR_LOSS = (
    COMPARISON_OUTPUT_DIR
    / "attention_discriminator_losses.npy"
)

ATTENTION_BASELINE_SAMPLES = (
    COMPARISON_OUTPUT_DIR
    / "baseline_gan_samples.pt"
)

ATTENTION_SAMPLES = (
    COMPARISON_OUTPUT_DIR
    / "attention_gan_samples.pt"
)

ATTENTION_WEIGHTS = (
    COMPARISON_OUTPUT_DIR
    / "attention_weights.pt"
)


# ============================================================
# APPLICATION DEFAULTS
# ============================================================

APP_TITLE = (
    "Text-to-Image Generation Studio"
)

APP_ICON = "🖼️"

DEFAULT_SEED = 42

DEFAULT_PROMPT = (
    "A blue tree in a forest at sunset"
)

DEFAULT_DEVICE = "cuda"


# ============================================================
# MODEL MODES
# ============================================================

MODE_REALISTIC = "realistic"

MODE_INTERNSHIP_LORA = "internship_lora"

MODE_RESEARCH = "research"


# ============================================================
# MODE DISPLAY NAMES
# ============================================================

MODE_REALISTIC_LABEL = (
    "Realistic Studio"
)

MODE_INTERNSHIP_LORA_LABEL = (
    "Internship Fine-Tuned"
)

MODE_RESEARCH_LABEL = (
    "Research Lab"
)


# ============================================================
# PATH CHECKS
# ============================================================

def task1_lora_available() -> bool:
    return TASK1_LORA_PATH.is_file()


def task6_artifacts_available() -> bool:

    return (
        TASK6_GENERATOR_PATH.is_file()
        and TASK6_DISCRIMINATOR_PATH.is_file()
        and TASK6_EMBEDDINGS_PATH.is_file()
    )


def attention_artifacts_available() -> bool:

    required_files = [
        ATTENTION_BASELINE_GENERATOR_LOSS,
        ATTENTION_GENERATOR_LOSS,
        ATTENTION_BASELINE_DISCRIMINATOR_LOSS,
        ATTENTION_DISCRIMINATOR_LOSS,
        ATTENTION_BASELINE_SAMPLES,
        ATTENTION_SAMPLES,
        ATTENTION_WEIGHTS,
    ]

    return all(
        path.is_file()
        for path in required_files
    )


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

def configure_page() -> None:

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )