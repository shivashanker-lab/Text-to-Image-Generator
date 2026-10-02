# Experiment Outputs

This directory contains selected artifacts produced by the project experiments.

## Comparisons

`outputs/comparisons/` contains saved results from the baseline GAN and the attention-conditioned GAN experiment:

- `baseline_generator_losses.npy`
- `baseline_discriminator_losses.npy`
- `baseline_gan_samples.pt`
- `attention_generator_losses.npy`
- `attention_discriminator_losses.npy`
- `attention_gan_samples.pt`
- `attention_weights.pt`

The saved GAN sample tensors have shape `[8, 1, 64, 64]`.

The attention-weight tensor has shape `[8, 16, 5]`.

## Graphs and Analysis

`outputs/graphs/flickr8k_analysis_results.csv` contains the summarized Flickr8k dataset-analysis metrics.

## Generated Images

`outputs/generated_images/` is reserved for generated image outputs.

The current local checkout does not contain generated image files in this directory because several experiments were executed in notebook or Colab runtimes and their runtime-generated image files were not retained in this repository checkout.

The notebooks contain the code used to generate and verify those images.
