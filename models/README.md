# Model Artifacts

This directory contains saved Conditional GAN artifacts from the internship project.

## Conditional GAN Artifacts

The `models/cgan/` directory contains:

- `task6_text_conditioned_generator.pth` - trained generator checkpoint.
- `task6_text_conditioned_discriminator.pth` - trained discriminator checkpoint.
- `task6_text_embeddings.pt` - saved text embeddings associated with the research pipeline.
- `task6_generator_losses.npy` - saved generator training-loss values.
- `task6_discriminator_losses.npy` - saved discriminator training-loss values.

The existing Streamlit application uses the configured GAN artifacts for its Research Lab mode. The discriminator checkpoint and loss files are retained as research and training artifacts.

## Stable Diffusion LoRA

The internship fine-tuning weights are stored at:

`outputs/task1/lora_custom_dataset/pytorch_lora_weights.safetensors`

The application references this artifact for Internship Fine-Tuned mode.

## Pretrained Base Models

The complete pretrained Stable Diffusion and BERT weights are not stored in this directory. The application loads models through their configured model identifiers and the Hugging Face model-loading system.
