from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from app.services.model_service import ModelService


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
COMPARISON_DIR = OUTPUTS_DIR / "comparisons"


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_TITLE = "Text-to-Image Studio"
APP_ICON = "🖼️"

DEFAULT_PROMPT = (
    "A cinematic blue tree in a peaceful forest at sunset"
)

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, out of focus, "
    "soft focus, noisy, grainy, jpeg artifacts, distorted, "
    "deformed, bad anatomy, extra fingers, extra limbs, "
    "duplicate objects, malformed hands, cropped, "
    "watermark, text, logo, oversaturated"
)

EXAMPLE_PROMPTS = [
    "A cinematic blue tree in a peaceful forest at sunset",
    "A black sports car on a wet city road at night",
    "A child in a pink dress climbing a staircase",
    "A futuristic city skyline at night",
    "A detailed mountain landscape at sunrise",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODEL SERVICE
# ============================================================

@st.cache_resource
def get_model_service() -> ModelService:
    return ModelService()


# ============================================================
# IMAGE HELPERS
# ============================================================

def image_to_png_bytes(
    image: Image.Image,
) -> bytes:

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def image_to_base64(
    image: Image.Image,
) -> str:

    return base64.b64encode(
        image_to_png_bytes(image)
    ).decode("utf-8")


def normalize_image(
    image,
) -> Image.Image:

    if isinstance(image, Image.Image):

        return image

    return Image.fromarray(
        np.asarray(image)
    )


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session():

    if "current_page" not in st.session_state:

        st.session_state[
            "current_page"
        ] = "converter"


def clear_generated_result():

    for key in [
        "generated_image",
        "generation_details",
        "generation_time",
        "generation_mode",
        "generation_prompt",
    ]:

        st.session_state.pop(
            key,
            None,
        )


# ============================================================
# MODE CONFIGURATION
# ============================================================

def get_mode_configuration(
    selected_mode: str,
):

    if selected_mode == "Realistic Studio":

        return (
            "realistic",
            "Stable Diffusion",
            (
                "General-purpose high-quality image "
                "generation."
            ),
        )

    if selected_mode == "Internship Fine-Tuned":

        return (
            "internship_lora",
            "Stable Diffusion + LoRA",
            (
                "Your internship fine-tuned Stable "
                "Diffusion pipeline."
            ),
        )

    return (
        "research",
        "BERT + Conditional GAN",
        (
            "Experimental text-conditioned research "
            "pipeline."
        ),
    )


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .app-title {
        font-size: 2.7rem;
        font-weight: 750;
        letter-spacing: -0.035em;
        margin-bottom: 0.15rem;
    }

    .app-subtitle {
        color: #707070;
        font-size: 1.05rem;
        margin-bottom: 1.25rem;
    }

    .sidebar-heading {
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.58;
        margin-top: 0.7rem;
        margin-bottom: 0.5rem;
    }

    .generation-overlay {
        position: relative;
        width: 100%;
        overflow: hidden;
        border-radius: 14px;
        margin-top: 0.7rem;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(128,128,128,0.2);
    }

    .generation-overlay img {
        display: block;
        width: 100%;
        opacity: 0.32;
        filter: saturate(0.65);
    }

    .generation-overlay-label {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        padding: 0.7rem 1rem;
        border-radius: 999px;
        background: rgba(0,0,0,0.72);
        color: white;
        font-size: 0.95rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .quality-note {
        font-size: 0.86rem;
        opacity: 0.68;
        margin-top: 0.25rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FADED PREVIOUS IMAGE
# ============================================================

def show_generating_preview(
    placeholder,
    image: Image.Image,
):

    encoded = image_to_base64(
        image
    )

    placeholder.markdown(
        f"""
        <div class="generation-overlay">
            <img src="data:image/png;base64,{encoded}">
            <div class="generation-overlay-label">
                ✨ Creating your new image...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

def render_sidebar(
    service: ModelService,
):

    with st.sidebar:

        st.markdown(
            "## 🖼️ Text-to-Image"
        )

        st.caption(
            "Internship Project • Phase 7"
        )

        st.divider()

        # ----------------------------------------------------
        # MAIN APPLICATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="sidebar-heading">'
            'Generation'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "🖼️ Text-to-Image Converter",
            width="stretch",
            type=(
                "primary"
                if st.session_state["current_page"]
                == "converter"
                else "secondary"
            ),
        ):

            st.session_state[
                "current_page"
            ] = "converter"

            st.rerun()

        # ----------------------------------------------------
        # RESEARCH & DOCUMENTATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="sidebar-heading">'
            'Research & Documentation'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "🔍 Attention Explorer",
            width="stretch",
            type=(
                "primary"
                if st.session_state["current_page"]
                == "attention"
                else "secondary"
            ),
        ):

            st.session_state[
                "current_page"
            ] = "attention"

            st.rerun()

        if st.button(
            "📘 Project Information",
            width="stretch",
            type=(
                "primary"
                if st.session_state["current_page"]
                == "information"
                else "secondary"
            ),
        ):

            st.session_state[
                "current_page"
            ] = "information"

            st.rerun()

        st.divider()

        # ----------------------------------------------------
        # SYSTEM STATUS
        # ----------------------------------------------------

        st.markdown(
            '<div class="sidebar-heading">'
            'System Status'
            '</div>',
            unsafe_allow_html=True,
        )

        if service.device.type == "cuda":

            st.success(
                "GPU ready"
            )

        else:

            st.warning(
                "CPU mode"
            )

        if service.task1_lora_available:

            st.write(
                "✅ Internship LoRA"
            )

        else:

            st.write(
                "⚠️ Internship LoRA unavailable"
            )

        if service.task6_available:

            st.write(
                "✅ Research GAN"
            )

        else:

            st.write(
                "⚠️ Research GAN unavailable"
            )

        st.divider()

        if st.button(
            "Clear Generated Result",
            width="stretch",
        ):

            clear_generated_result()

            st.rerun()


# ============================================================
# TEXT-TO-IMAGE PAGE
# ============================================================

def render_converter(
    service: ModelService,
):

    st.markdown(
        '<div class="app-title">'
        'Text-to-Image Converter'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Describe an idea and transform it into an image.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ========================================================
    # MODEL
    # ========================================================

    st.markdown(
        "### Generation Model"
    )

    selected_mode = st.radio(
        "Generation model",
        [
            "Realistic Studio",
            "Internship Fine-Tuned",
            "Research Lab",
        ],
        horizontal=True,
    )

    mode, technology, description = (
        get_mode_configuration(
            selected_mode
        )
    )

    st.info(
        f"**{technology}** · {description}"
    )

    # ========================================================
    # PROMPT
    # ========================================================

    st.markdown(
        "### Prompt"
    )

    example_choice = st.selectbox(
        "Example prompts",
        [
            "Custom prompt",
            *EXAMPLE_PROMPTS,
        ],
        label_visibility="collapsed",
    )

    if (
        example_choice
        == "Custom prompt"
    ):

        initial_prompt = (
            st.session_state.get(
                "prompt_value",
                DEFAULT_PROMPT,
            )
        )

    else:

        initial_prompt = example_choice

    prompt = st.text_area(
        "Describe your image",
        value=initial_prompt,
        height=115,
        placeholder=(
            "Example: A cinematic portrait "
            "of a traveler in the mountains..."
        ),
        label_visibility="collapsed",
    )

    st.session_state[
        "prompt_value"
    ] = prompt

    # ========================================================
    # QUALITY CONTROLS
    # ========================================================

    with st.expander(
        "⚙️ Quality Controls",
        expanded=True,
    ):

        col1, col2 = st.columns(2)

        with col1:

            steps = st.slider(
                "Inference Steps",
                min_value=25,
                max_value=50,
                value=35,
                step=1,
                disabled=(
                    mode == "research"
                ),
                help=(
                    "More steps can improve refinement "
                    "but increase generation time."
                ),
            )

        with col2:

            guidance = st.slider(
                "Guidance Scale",
                min_value=5.0,
                max_value=9.0,
                value=7.5,
                step=0.5,
                disabled=(
                    mode == "research"
                ),
                help=(
                    "Controls how strongly the image "
                    "follows the prompt."
                ),
            )

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
        )

        if mode == "research":

            negative_prompt = ""

            st.caption(
                "Research Lab uses the trained GAN "
                "and does not use diffusion settings."
            )

        else:

            negative_prompt = st.text_area(
                "Negative Prompt",
                value=DEFAULT_NEGATIVE_PROMPT,
                height=90,
                help=(
                    "Unwanted visual characteristics "
                    "to avoid."
                ),
            )

            st.markdown(
                '<div class="quality-note">'
                'The default quality controls are tuned '
                'for Stable Diffusion v1.5.'
                '</div>',
                unsafe_allow_html=True,
            )

    # ========================================================
    # GENERATE
    # ========================================================

    st.markdown(
        "### Generate"
    )

    generate_clicked = st.button(
        "✨ Generate Image",
        type="primary",
        width="stretch",
    )

    if generate_clicked:

        clean_prompt = prompt.strip()

        if not clean_prompt:

            st.warning(
                "Please enter a prompt before generating."
            )

            return

        # Keep the previous image.
        previous_image = st.session_state.get(
            "generated_image"
        )

        # Placeholder used only while the model is running.
        previous_preview = st.empty()

        if previous_image is not None:

            show_generating_preview(
                previous_preview,
                previous_image,
            )

        started = time.perf_counter()

        try:

            with st.spinner(
                "Generating your new image..."
            ):

                result = service.generate(
                    prompt=clean_prompt,
                    mode=mode,
                    seed=int(seed),
                    steps=int(steps),
                    guidance_scale=float(
                        guidance
                    ),
                    negative_prompt=(
                        negative_prompt.strip()
                    ),
                )

            elapsed = (
                time.perf_counter()
                - started
            )

            if not isinstance(
                result,
                dict,
            ):

                raise RuntimeError(
                    "The generation service returned "
                    "an invalid result."
                )

            if "image" not in result:

                raise RuntimeError(
                    "The generation service returned "
                    "no image."
                )

            image = normalize_image(
                result["image"]
            )

            # Remove faded old preview.
            previous_preview.empty()

            # Store the new image.
            st.session_state[
                "generated_image"
            ] = image

            st.session_state[
                "generation_details"
            ] = result

            st.session_state[
                "generation_time"
            ] = elapsed

            st.session_state[
                "generation_mode"
            ] = selected_mode

            st.session_state[
                "generation_prompt"
            ] = clean_prompt

            st.success(
                f"Image generated successfully "
                f"in {elapsed:.1f} seconds."
            )

        except torch.cuda.OutOfMemoryError:

            previous_preview.empty()

            try:

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

            except Exception:

                pass

            st.error(
                "The GPU ran out of memory. "
                "Try reducing the inference steps."
            )

            with st.expander(
                "Technical Details"
            ):

                st.code(
                    traceback.format_exc(),
                    language="text",
                )

        except Exception:

            previous_preview.empty()

            st.error(
                "The image could not be generated."
            )

            with st.expander(
                "Technical Details"
            ):

                st.code(
                    traceback.format_exc(),
                    language="text",
                )

    # ========================================================
    # RESULT
    # ========================================================

    render_result()


# ============================================================
# GENERATED RESULT
# ============================================================

def render_result():

    if (
        "generated_image"
        not in st.session_state
    ):

        return

    image = (
        st.session_state[
            "generated_image"
        ]
    )

    details = (
        st.session_state.get(
            "generation_details",
            {},
        )
    )

    elapsed = (
        st.session_state.get(
            "generation_time",
            0.0,
        )
    )

    selected_mode = (
        st.session_state.get(
            "generation_mode",
            "Unknown",
        )
    )

    prompt = (
        st.session_state.get(
            "generation_prompt",
            "",
        )
    )

    st.divider()

    st.markdown(
        "### Generated Result"
    )

    image_col, detail_col = st.columns(
        [2.25, 1],
        gap="large",
    )

    with image_col:

        st.image(
            image,
            caption="Generated Image",
            width="stretch",
        )

    with detail_col:

        st.markdown(
            "#### Generation Details"
        )

        st.markdown(
            "**Model**"
        )

        st.write(
            selected_mode
        )

        st.markdown(
            "**Prompt**"
        )

        st.write(
            prompt
        )

        st.markdown(
            "**Resolution**"
        )

        st.write(
            f"{image.size[0]} × {image.size[1]}"
        )

        if details.get(
            "seed"
        ) is not None:

            st.markdown(
                "**Seed**"
            )

            st.write(
                str(
                    details[
                        "seed"
                    ]
                )
            )

        if details.get(
            "steps"
        ) is not None:

            st.markdown(
                "**Inference Steps**"
            )

            st.write(
                str(
                    details[
                        "steps"
                    ]
                )
            )

        if details.get(
            "guidance_scale"
        ) is not None:

            st.markdown(
                "**Guidance**"
            )

            st.write(
                str(
                    details[
                        "guidance_scale"
                    ]
                )
            )

        st.markdown(
            "**Device**"
        )

        st.write(
            details.get(
                "device",
                "Unknown",
            )
        )

        st.markdown(
            "**Generation Time**"
        )

        st.write(
            f"{elapsed:.1f} seconds"
        )

        if details.get(
            "lora_active",
            False,
        ):

            st.success(
                "LoRA active"
            )

        if details.get(
            "embedding_shape"
        ):

            st.write(
                "BERT embedding: "
                + str(
                    details[
                        "embedding_shape"
                    ]
                )
            )

        st.download_button(
            label="⬇️ Download PNG",
            data=image_to_png_bytes(
                image
            ),
            file_name=(
                "text_to_image_result.png"
            ),
            mime="image/png",
            width="stretch",
        )


# ============================================================
# ATTENTION EXPLORER
# ============================================================

def render_attention_explorer():

    st.markdown(
        '<div class="app-title">'
        'Attention Explorer'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Explore the attention experiment and '
        'saved comparison artifacts.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    baseline_path = (
        COMPARISON_DIR
        / "baseline_generator_losses.npy"
    )

    attention_path = (
        COMPARISON_DIR
        / "attention_generator_losses.npy"
    )

    if (
        baseline_path.is_file()
        and attention_path.is_file()
    ):

        try:

            baseline = np.load(
                baseline_path
            )

            attention = np.load(
                attention_path
            )

            length = min(
                len(baseline),
                len(attention),
            )

            chart_df = pd.DataFrame(
                {
                    "Baseline Generator": (
                        baseline[:length]
                    ),
                    "Attention Generator": (
                        attention[:length]
                    ),
                }
            )

            st.markdown(
                "### Generator Loss Comparison"
            )

            st.line_chart(
                chart_df,
                width="stretch",
            )

        except Exception:

            st.warning(
                "The saved loss files could not "
                "be displayed."
            )

    else:

        st.info(
            "Attention comparison files are not available."
        )

    st.divider()

    st.markdown(
        "### Experiment Artifacts"
    )

    artifacts = [
        "baseline_generator_losses.npy",
        "attention_generator_losses.npy",
        "baseline_discriminator_losses.npy",
        "attention_discriminator_losses.npy",
        "baseline_gan_samples.pt",
        "attention_gan_samples.pt",
        "attention_weights.pt",
    ]

    for filename in artifacts:

        path = (
            COMPARISON_DIR
            / filename
        )

        if path.is_file():

            st.write(
                f"✅ {filename}"
            )

        else:

            st.write(
                f"⚠️ {filename}"
            )


# ============================================================
# PROJECT INFORMATION
# ============================================================

def render_project_information():

    st.markdown(
        '<div class="app-title">'
        'Project Information'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-subtitle">'
        'The internship work behind the application.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        """
### Internship Pipeline

**01 · Pretrained Model + LoRA**
Stable Diffusion adaptation using Low-Rank Adaptation.

**02 · Conditional GAN**
Text-conditioned image-generation experiment.

**03 · Text Processing + BERT**
Caption preprocessing and transformer embeddings.

**04 · Dataset Analysis**
Image-caption dataset analysis.

**05 · Attention Mechanism**
Attention-based GAN experiment.

**06 · Integrated Pipeline**
Text → preprocessing → BERT → Conditional GAN → image.
"""
    )

    st.divider()

    st.markdown(
        "### Application Architecture"
    )

    architecture = pd.DataFrame(
        {
            "Mode": [
                "Realistic Studio",
                "Internship Fine-Tuned",
                "Research Lab",
            ],
            "Technology": [
                "Stable Diffusion",
                "Stable Diffusion + LoRA",
                "BERT + Conditional GAN",
            ],
            "Purpose": [
                "Realistic image generation",
                "Demonstrate internship fine-tuning",
                "Demonstrate the research pipeline",
            ],
        }
    )

    st.table(architecture)

    st.divider()

    st.markdown(
        "### Model Note"
    )

    st.write(
        "Stable Diffusion provides the main realistic "
        "generation experience. The LoRA pipeline demonstrates "
        "the custom internship adaptation. The Conditional "
        "GAN remains the research-focused pipeline because "
        "it was trained for the internship experiment and "
        "operates at a lower resolution."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    initialize_session()

    service = get_model_service()

    render_sidebar(
        service
    )

    current_page = (
        st.session_state[
            "current_page"
        ]
    )

    if current_page == "converter":

        render_converter(
            service
        )

    elif current_page == "attention":

        render_attention_explorer()

    elif current_page == "information":

        render_project_information()

    st.divider()

    st.caption(
        "Internship Project • Phase 7 • "
        "Text-to-Image Generation Studio"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()