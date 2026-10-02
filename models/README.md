# Model Artifacts

## Repository Model Folders

- `models/cgan/` — reserved for conditional GAN checkpoints.
- `models/pretrained/` — reserved for pretrained or adapted model artifacts.

Large model files are not committed to this repository.

## Task 1: Stable Diffusion LoRA

The Task 1 notebook saves the trained LoRA weights as:

`outputs/task1/lora_custom_dataset/pytorch_lora_weights.safetensors`

The completed training notebook verified that the weights were saved and successfully reloaded into a fresh Stable Diffusion pipeline. The trained LoRA file is not currently present in this local repository checkout.

## Task 6: Integrated Text-to-Image Pipeline

The Task 6 notebook saves:

- `task6_text_conditioned_generator.pth`
- `task6_text_conditioned_discriminator.pth`
- `task6_text_embeddings.pt`

These were generated in the notebook runtime under:

`/content/text_to_image_task6/`

They are not currently stored in this repository checkout.

## Storage Strategy

Large model checkpoints are kept outside GitHub to avoid committing large binary artifacts. The notebooks contain the code required to create, save, reload, and verify the trained model artifacts.
