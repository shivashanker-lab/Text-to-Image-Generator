# Project Scripts

This directory contains scripts for reproducing selected project experiments.

## `reproduce_task1_lora.py`

### Purpose
Reproduces the Task 1 Stable Diffusion LoRA fine-tuning experiment using image-caption examples from the Flickr8k dataset.

### Workflow
The script prepares training examples, loads the Stable Diffusion pipeline, generates baseline images, trains LoRA parameters, saves the LoRA weights and training-loss plot, reloads the saved adapter, generates adapted images, and creates a baseline-versus-adapted comparison.

### Configuration
The current script is configured for 60 training examples, 3 epochs, batch size 1, and a learning rate of `5e-5`. It uses the model ID `stable-diffusion-v1-5/stable-diffusion-v1-5` and dataset ID `intro/flickr8k`.

### Outputs
The script writes experiment artifacts under `outputs/task1/`, including baseline images, adapted images, training results, LoRA weights, and a comparison image.

**Important:** Running this script may overwrite existing files in those output directories. Back up any artifacts you need to preserve before running it.

### Requirements and validation
Run the script using the project's configured Python environment with its required dependencies installed. Dataset access may require an internet connection, and model availability depends on the local cache and configuration. A compatible CUDA GPU is recommended for this training workload.

The script has passed a Python syntax compilation check. A complete end-to-end training run has not been confirmed as part of this folder-organization review.
