# Data Organization

## Dataset

This project uses the public Flickr8k image-caption dataset.

Hugging Face dataset ID:

`intro/flickr8k`

The dataset is loaded through the Hugging Face `datasets` library in streaming mode, so the full dataset is not stored inside this repository.

## Repository Folders

- `data/raw/` — reserved for raw dataset files when local copies are required.
- `data/processed/` — reserved for processed or transformed datasets.

Both folders are currently empty because the project loads Flickr8k through streaming rather than storing the dataset locally.

## Dataset Usage

The dataset is used for:

- dataset analysis
- text preprocessing and embedding experiments
- Stable Diffusion LoRA fine-tuning experiments

See the notebooks in `notebooks/` for the complete analysis and experiments.
