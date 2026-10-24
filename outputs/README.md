# Experiment Outputs

This directory contains saved experiment results, generated images, and model artifacts.

## GAN Comparisons

The `comparisons/` directory preserves baseline and attention-based GAN results:

- `baseline_generator_losses.npy`
- `baseline_discriminator_losses.npy`
- `baseline_gan_samples.pt`
- `attention_generator_losses.npy`
- `attention_discriminator_losses.npy`
- `attention_gan_samples.pt`
- `attention_weights.pt`

The loss arrays record five values per experiment. The sample tensors and attention weights are retained for inspection and reproducibility.

## Dataset Analysis

`graphs/flickr8k_analysis_results.csv` contains summarized Flickr8k dataset statistics, including image counts, caption statistics, and image dimensions.

## Generated Images

The `generated_images/` directory contains these validated showcase images:

- `realistic_studio.png`
- `internship_finetuned.png`

## Task 1: Stable Diffusion and LoRA

The `task1/` directory contains:

- `baseline/baseline_1.png`, `baseline/baseline_2.png`, and `baseline/baseline_3.png`
- `adapted/adapted_1.png`, `adapted/adapted_2.png`, and `adapted/adapted_3.png`
- `comparison/baseline_vs_adapted_comparison.png`
- `training/training_loss.png`
- `lora_custom_dataset/pytorch_lora_weights.safetensors`

The consolidated comparison board presents all three baseline/adapted pairs and their amplified pixel differences.

The baseline and adapted outputs are not pixel-identical. However, a pixel difference alone does not establish that fine-tuning improved visual quality. Pair 2 also used slightly different prompts, so that pair is not a fully controlled comparison.

## Reproducibility

The project's numbered notebooks contain the associated experiments and code. They are retained unchanged, as requested.
