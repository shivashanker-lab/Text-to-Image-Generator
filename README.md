# Text-to-Image Generation Studio

### Data Science Internship Project — ElevanceSkills

A modular text-to-image generation research project integrating pretrained diffusion models, Low-Rank Adaptation (LoRA), Transformer-based text representations, Conditional Generative Adversarial Networks (CGANs), attention mechanisms, and public image-caption dataset analysis.

The project explores how natural-language descriptions can be transformed into meaningful numerical representations and used to condition generative models. It includes six internship tasks, experimental notebooks, reusable Python implementations, trained model artifacts, evaluation results, and a Streamlit-based application.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Objectives](#2-project-objectives)
3. [Internship Tasks](#3-internship-tasks)
4. [Application Features](#4-application-features)
5. [System Architecture](#5-system-architecture)
6. [Repository Structure](#6-repository-structure)
7. [Technology Stack](#7-technology-stack)
8. [Development Environment](#8-development-environment)
9. [Installation and Setup](#9-installation-and-setup)
10. [Running the Application](#10-running-the-application)
11. [Model Artifacts and Experimental Results](#11-model-artifacts-and-experimental-results)
12. [Dataset Information](#12-dataset-information)
13. [Reproducibility](#13-reproducibility)
14. [Verification and Testing](#14-verification-and-testing)
15. [Limitations and Considerations](#15-limitations-and-considerations)
16. [Troubleshooting](#16-troubleshooting)
17. [Internship Deliverable Coverage](#17-internship-deliverable-coverage)
18. [Conclusion](#18-conclusion)

---

## 1. Project Overview

Text-to-Image Generation Studio is an educational and research-oriented project developed as part of the ElevanceSkills Data Science Internship.

The project investigates different approaches to generating images from natural-language descriptions. It combines traditional generative adversarial networks, text-conditioned generation, attention-based architectures, and pretrained diffusion models.

The project consists of six interconnected internship tasks:

- Fine-tuning a pretrained Stable Diffusion model using LoRA and a curated image-caption dataset.
- Implementing a Conditional Generative Adversarial Network for label-conditioned image generation.
- Processing natural-language descriptions using Hugging Face Transformers and BERT.
- Exploring and analyzing the Flickr8k public image-caption dataset.
- Implementing attention mechanisms for text-conditioned GAN generation.
- Integrating text preprocessing, embeddings, and GAN-based image generation into a comprehensive pipeline.

The project also includes a Streamlit interface that provides access to selected image-generation models and experimental demonstrations.

### Key Project Components

| Component | Description |
|---|---|
| Stable Diffusion | Pretrained text-to-image generation |
| LoRA | Parameter-efficient fine-tuning |
| Conditional GAN | Label-conditioned synthetic image generation |
| BERT | Contextual text representation |
| Cross-Attention | Text-conditioned visual feature processing |
| Flickr8k | Public image-caption dataset exploration |
| Integrated Pipeline | Text processing and GAN-based image generation |
| Streamlit | Interactive application interface |

The project maintains separate experimental workflows for diffusion-based generation and GAN-based generation. These approaches have different architectures, objectives, and output capabilities.

---

## 2. Project Objectives

The primary objectives are:

1. Understand the relationship between natural-language descriptions and image generation.
2. Explore and analyze a public image-caption dataset.
3. Preprocess text descriptions and convert them into numerical embeddings.
4. Implement conditional generative adversarial networks.
5. Investigate attention mechanisms for improving text-conditioned visual generation.
6. Adapt a pretrained Stable Diffusion model using Low-Rank Adaptation.
7. Integrate text preprocessing, embedding creation, and GAN-based generation.
8. Preserve trained checkpoints, generated images, and experimental evaluation results.
9. Develop a modular application for demonstrating the implemented approaches.
10. Maintain reproducible experiments and clear technical documentation.

---

## 3. Internship Tasks

The project is organized around the six assigned ElevanceSkills internship tasks.

### Task 1 — Fine-Tuning a Pretrained Text-to-Image Model

**Objective:** Adapt a pretrained text-to-image model using a custom image-caption subset to explore domain-specific generation.

**Implementation:**

The project uses Stable Diffusion 1.5 with Low-Rank Adaptation (LoRA).

The fine-tuning workflow includes:

1. Loading the Flickr8k image-caption dataset.
2. Filtering animal-related descriptions to construct a curated training subset.
3. Selecting 60 image-caption examples for the configured demonstration experiment.
4. Loading the pretrained Stable Diffusion pipeline.
5. Configuring LoRA adapters in the UNet attention projections.
6. Freezing the base model parameters while optimizing the adapter parameters.
7. Computing diffusion noise-prediction loss.
8. Recording training losses.
9. Saving adapter weights in Safetensors format.
10. Reloading the saved adapter and generating adapted image examples.

The reproduction script is:

` scripts/reproduce_task1_lora.py `

The reusable implementation is:

`src/models/stable_diffusion_lora.py`

**Training configuration recorded in the reproduction workflow:**

| Parameter | Configuration |
|---|---:|
| Base model | Stable Diffusion 1.5 |
| Dataset | Curated Flickr8k animal subset |
| Training examples | 60 |
| Epochs | 3 |
| Batch size | 1 |
| Learning rate | 5e-5 |
| Total configured training steps | 180 |
| LoRA rank | Defined in training configuration |
| Text conditioning | Stable Diffusion text encoder |

**Main trained artifact:**

`outputs/task1/lora_custom_dataset/pytorch_lora_weights.safetensors`

**Additional experimental outputs:**

- Baseline image samples.
- LoRA-adapted image samples.
- Training-loss visualization.
- Baseline-versus-adapted comparison image.

The saved LoRA checkpoint was verified as readable and contained 256 finite tensors during the project verification.

The presence of different baseline and adapted outputs does not automatically demonstrate improved image quality. Comparisons must use equivalent prompts, seeds, and generation settings.

### Task 2 — Conditional Generative Adversarial Network

**Objective:** Implement a Conditional GAN that generates basic visual shapes according to supplied category labels.

**Implementation:**

The project implements separate generator and discriminator networks.

The generator receives:

- Random latent noise.
- A learned embedding of the conditional class label.

The discriminator receives:

- An input image.
- A spatial representation of the supplied class label.

The two supported classes are:

| Class label | Shape |
|---|---|
| 0 | Square |
| 1 | Circle |

**Implementation file:**

`src/models/conditional_gan.py`

**Experiment notebook:**

`notebooks/03_conditional_gan.ipynb`

The generator produces 64 × 64 grayscale images.

The discriminator evaluates the relationship between the supplied condition and the generated image.

**Verification:**

A model forward-pass test successfully generated an output tensor with the shape:

`(2, 1, 64, 64)`

The test used two different conditional labels.

This experiment demonstrates label-conditioned image generation using a GAN architecture. It is a synthetic-shape research model rather than a general-purpose photorealistic image generator.

### Task 3 — Text Preprocessing and BERT Embeddings

**Objective:** Transform text descriptions into numerical representations suitable for conditioning generative models.

**Implementation:**

The project uses Hugging Face Transformers and the pretrained BERT model:

`bert-base-uncased`

The preprocessing workflow includes:

1. Lowercasing text.
2. Removing punctuation.
3. Normalizing whitespace.
4. Tokenizing descriptions.
5. Creating token IDs and attention masks.
6. Extracting contextual token representations using BERT.
7. Applying attention-mask-aware mean pooling.
8. Producing sentence-level embeddings.

**Implementation file:**

`src/preprocessing/text_embeddings.py`

**Main configuration:**

| Parameter | Value |
|---|---|
| Text encoder | BERT Base Uncased |
| Maximum sequence length | 32 |
| Embedding dimension | 768 |
| Pooling | Masked mean pooling |
| Framework | Hugging Face Transformers and PyTorch |

The implementation automatically moves tokenized inputs to the same device as the text encoder.

The resulting embedding can be passed to a downstream text-conditioned generative model.

### Task 4 — Public Dataset Exploration and Analysis

**Objective:** Understand the structure and statistical characteristics of a public image-caption dataset.

**Dataset:** Flickr8k

**Hugging Face dataset identifier:**

`intro/flickr8k`

The project uses the Hugging Face Datasets library to access the dataset, including streaming mode where appropriate.

**Implementation file:**

`src/data/flickr8k_loader.py`

**Experiment notebook:**

`notebooks/01_dataset_analysis.ipynb`

The dataset-analysis workflow explores:

- Dataset organization and available splits.
- Image-caption relationships.
- Number of images represented in the analyzed dataset.
- Caption distributions.
- Description lengths.
- Image characteristics and dimensions.
- Visual examples of images with their corresponding descriptions.

**Saved analysis artifact:**

`outputs/graphs/flickr8k_analysis_results.csv`

The associated notebook contains the analysis workflow and visual exploration.

The exact interpretation of the saved statistics depends on the selected dataset split and the counting procedures used during analysis.

### Task 5 — Attention Mechanisms in GANs

**Objective:** Implement attention-based conditioning to enable visual features to focus on relevant text representations.

**Implementation:**

The project includes baseline and attention-based text-conditioned GAN architectures.

The attention model uses cross-attention:

- Visual features provide queries.
- BERT token features provide keys.
- BERT token features provide values.
- Attention weights determine how text information contributes to visual conditioning.

The implementation also includes:

- Learned query, key, and value projections.
- Scaled dot-product attention.
- Padding-mask handling.
- Residual conditioning.
- Normalization layers.
- A baseline model without cross-attention.

**Reusable architecture files:**

`src/models/attention_gan.py`

`src/models/task5_attention_gan_model.py`

**Task 5 model artifacts:**

`models/cgan/task5/`

The Task 5 experiment uses the following configuration recorded in its model configuration file:

| Parameter | Value |
|---|---:|
| Image size | 32 × 32 |
| Image channels | 1 |
| Latent dimension | 100 |
| Text embedding dimension | 768 |
| Visual dimension | 128 |
| Attention dimension | 128 |
| Maximum text length | 16 |
| Training epochs recorded | 50 |
| Batch size recorded | 32 |

The task is focused on generating simple grayscale square and circle shapes.

**Saved experiment artifacts include:**

- Attention generator checkpoint.
- Attention discriminator checkpoint.
- Baseline generator checkpoint.
- Baseline discriminator checkpoint.
- Model configuration.
- Training curves.
- Evaluation metrics.
- Validation report.
- Generated validation images.

**Selected recorded validation results:**

| Evaluation metric | Baseline | Attention |
|---|---:|---:|
| Square condition accuracy | 1.00 | 1.00 |
| Circle condition accuracy | 1.00 | 1.00 |
| Mean shape IoU | 1.00 | 1.00 |

The metrics were recorded using 24 generated examples per prompt.

The validation report records successful strict checkpoint loading, valid output dimensions, and finite pixel values.

Both models achieved the same reported shape-accuracy and IoU results. Therefore, these results demonstrate successful validation of the implemented attention experiment but do not establish that attention outperforms the baseline.

**Canonical Task 5 artifacts:**

`models/cgan/task5/`

**Additional validation images:**

`outputs/task5/validation/`

**Portable experiment package:**

`outputs/task5/colab_bundle/task5_artifacts/`

The Colab bundle is retained as a portable copy of the Task 5 experiment artifacts.

### Task 6 — Integrated Text-to-Image Generation Pipeline

**Objective:** Integrate text preprocessing, Transformer-based embeddings, and GAN-based generation into a comprehensive text-to-image workflow.

**Experiment notebook:**

`notebooks/06_integrated_text_to_image_pipeline.ipynb`

The conceptual pipeline is:

```text
Natural-Language Prompt
          |
          v
    Text Cleaning
          |
          v
   BERT Tokenization
          |
          v
 Contextual Text Features
          |
          v
 Text-Conditioned GAN
          |
          v
   Image Generation
          |
          v
  Generated Image Output
```

The integrated research workflow uses:

- Natural-language input.
- Text preprocessing.
- BERT contextual representations.
- Random latent vectors.
- A text-conditioned GAN generator.
- Image output processing.

**Associated Task 6 artifacts:**

| File | Purpose |
|---|---|
| `models/cgan/task6_text_conditioned_generator.pth` | Generator checkpoint |
| `models/cgan/task6_text_conditioned_discriminator.pth` | Discriminator checkpoint |
| `models/cgan/task6_text_embeddings.pt` | Saved text embeddings |
| `models/cgan/task6_generator_losses.npy` | Generator training-loss values |
| `models/cgan/task6_discriminator_losses.npy` | Discriminator training-loss values |

The Task 6 generator checkpoint was successfully loaded through the model service during verification. Its forward pass generated finite outputs of shape:

`(2, 1, 64, 64)`

Task 6 is a separate experimental integration workflow and should not be confused with the current Research Lab application route, which uses the Task 5 attention generator.

---

## 4. Application Features

The Streamlit application provides a unified interface for selected text-to-image and research demonstrations.

| Application mode | Model or component | Purpose |
|---|---|---|
| Realistic Studio | Stable Diffusion | General-purpose text-to-image generation |
| Internship Fine-Tuned | Stable Diffusion + LoRA | Demonstration of the fine-tuned adapter |
| Research Lab | Task 5 Attention GAN | Experimental text-conditioned shape generation |
| Attention Explorer | Saved GAN experiment results | Inspection of baseline and attention experiments |
| Project Information | Project documentation | Overview of the project and its components |

### Application capabilities

- Natural-language prompt input.
- Image generation using supported model pipelines.
- Seed configuration.
- Inference-step controls where supported.
- Guidance-scale controls where supported.
- Negative-prompt controls for diffusion modes.
- Image preview.
- PNG image download.
- Generation information.
- Device information.
- Generation-time reporting.
- Inspection of saved attention-experiment results.

The available controls and behavior depend on the selected model.

**Application entry point:**

`run_app.py`

**Main interface:**

`app/app.py`

**Model service:**

`app/services/model_service.py`

---

## 5. System Architecture

The project separates the application interface, reusable model implementations, research notebooks, trained checkpoints, and experimental outputs.

### High-level architecture

```text
                    User Prompt
                        |
                        v
              Streamlit Application
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
     Realistic     Internship     Research
      Studio       Fine-Tuned       Lab
          |             |             |
          v             v             v
      Stable        Stable        Attention
     Diffusion     Diffusion        GAN
                        |
                        v
                   LoRA Adapter


              Supporting Research
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
       Dataset       Text          GAN and
       Analysis    Embeddings      Attention
                                    Research
                        |
                        v
              Experimental Artifacts
```

### Architectural responsibilities

**Application Layer — `app/`**

Provides the Streamlit interface, configuration, and model-serving functionality.

**Data Layer — `data/` and `src/data/`**

Contains dataset documentation and reusable dataset-loading code.

**Preprocessing Layer — `src/preprocessing/`**

Contains text normalization and BERT embedding generation.

**Model Layer — `src/models/`**

Contains the Conditional GAN, attention mechanisms, and Stable Diffusion LoRA implementation.

**Experiment Layer — `notebooks/`**

Contains the six task-specific research notebooks and environment setup notebook.

**Model Artifacts — `models/`**

Contains selected GAN checkpoints and model configurations.

**Experiment Outputs — `outputs/`**

Contains generated images, LoRA weights, training curves, attention comparisons, analysis results, and validation artifacts.

**Automation and Reproduction — `scripts/`**

Contains scripts for reproducing selected experiments.

---

## 6. Repository Structure

The repository is organized as follows:

```text
Text-to-Image-Generator/
│
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   └── services/
│       ├── __init__.py
│       └── model_service.py
│
├── data/
│   └── README.md
│
├── models/
│   ├── README.md
│   └── cgan/
│       ├── task6_text_conditioned_generator.pth
│       ├── task6_text_conditioned_discriminator.pth
│       ├── task6_text_embeddings.pt
│       ├── task6_generator_losses.npy
│       ├── task6_discriminator_losses.npy
│       └── task5/
│           ├── task5_attention_config.json
│           ├── task5_attention_discriminator.pth
│           ├── task5_attention_generator.pth
│           ├── task5_baseline_discriminator.pth
│           ├── task5_baseline_generator.pth
│           ├── task5_attention_gan_model.py
│           ├── task5_generation_validation.png
│           ├── task5_metrics.json
│           ├── task5_training_curves.png
│           └── task5_validation_report.json
│
├── notebooks/
│   ├── 00_environment_setup.ipynb
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_text_preprocessing_embeddings.ipynb
│   ├── 03_conditional_gan.ipynb
│   ├── 04_attention_mechanism.ipynb
│   ├── 05_pretrained_model_fine_tuning.ipynb
│   └── 06_integrated_text_to_image_pipeline.ipynb
│
├── outputs/
│   ├── README.md
│   ├── comparisons/
│   ├── generated_images/
│   ├── graphs/
│   ├── task1/
│   │   ├── adapted/
│   │   ├── baseline/
│   │   ├── comparison/
│   │   ├── lora_custom_dataset/
│   │   └── training/
│   └── task5/
│       ├── colab_bundle/
│       │   └── task5_artifacts/
│       └── validation/
│
├── reports/
│   └── Text_to_Image_Generator_ElevanceSkills_Final_Internship_Report.docx
│
├── scripts/
│   ├── README.md
│   └── reproduce_task1_lora.py
│
├── src/
│   ├── data/
│   │   └── flickr8k_loader.py
│   ├── models/
│   │   ├── attention_gan.py
│   │   ├── conditional_gan.py
│   │   ├── stable_diffusion_lora.py
│   │   └── task5_attention_gan_model.py
│   ├── preprocessing/
│   │   └── text_embeddings.py
│   └── utils/
│       └── image_utils.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── run_app.py
```

This tree represents the intended final submission layout. It excludes local-only files such as virtual environments, Python caches, temporary files, and backup copies.

The Task 5 Colab bundle is intentionally retained as a portable experiment package. Some files in it are identical to canonical Task 5 artifacts.

---

## 7. Technology Stack

| Technology | Role |
|---|---|
| Python | Main programming language |
| PyTorch | Deep-learning framework |
| Streamlit | Interactive application interface |
| Hugging Face Diffusers | Stable Diffusion pipeline |
| Hugging Face Transformers | Tokenization and BERT |
| PEFT | Parameter-efficient fine-tuning |
| LoRA | Stable Diffusion adaptation |
| BERT | Contextual text representations |
| Conditional GAN | Conditional image generation |
| NumPy | Numerical experiment data |
| Pandas | Dataset analysis |
| Pillow | Image processing |
| Matplotlib | Result visualization |
| Hugging Face Datasets | Public dataset access |
| Safetensors | LoRA weight serialization |
| Jupyter Notebook | Experiment development |

---

## 8. Development Environment

The local development environment recorded for the project is:

| Component | Installed version or hardware |
|---|---|
| Operating system | Windows |
| Python | 3.12.10 |
| PyTorch | 2.9.1+cu130 |
| Diffusers | 0.40.0 |
| Transformers | 5.18.0 |
| Accelerate | 1.15.0 |
| Streamlit | 1.64.0 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU memory | 6 GB |
| CUDA availability | True |

The project uses GPU acceleration where supported.

### Dependency compatibility note

The current `requirements.txt` is the dependency manifest for the repository. Its declared version bounds should be reconciled with the installed development environment before a fresh installation is advertised as fully reproducible.

In particular, the installed Transformers and Pandas versions observed during development are outside the upper bounds currently declared in the requirements file.

Do not assume that a clean installation using the current manifest will reproduce the exact local environment until those constraints have been aligned and tested.

---

## 9. Installation and Setup

### Step 1 — Clone the repository

```powershell
git clone <YOUR_PUBLIC_GITHUB_REPOSITORY_URL>
```

Navigate into the project:

```powershell
cd Text-to-Image-Generator
```

### Step 2 — Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
```

Install the declared requirements:

```powershell
pip install -r requirements.txt
```

For GPU-enabled PyTorch, ensure the installed build is compatible with the available NVIDIA GPU and CUDA environment.

### Step 4 — Check PyTorch GPU availability

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )
```

### Step 5 — Configure pretrained model access

The project uses pretrained models from the Hugging Face ecosystem.

Depending on the selected model and local cache, internet access may be required to download pretrained weights.

The complete pretrained Stable Diffusion and BERT base weights are not bundled with this repository.

---

## 10. Running the Application

Execute the following command from the project root:

```powershell
python -m streamlit run .\run_app.py
```

To bind the local development server to the loopback interface:

```powershell
python -m streamlit run .\run_app.py --server.address 127.0.0.1
```

Streamlit will display the local application address in the terminal.

Open that address in a web browser while the application is running.

To stop the application, press:

```text
Ctrl + C
```

### Application Usage

**Realistic Studio**

Enter a natural-language prompt, configure the available diffusion settings, generate an image, and inspect or download the result.

**Internship Fine-Tuned**

Select the fine-tuned generation mode to use the configured Stable Diffusion base model with the saved internship LoRA adapter.

**Research Lab**

Use natural-language prompts to explore the experimental Task 5 attention-based GAN. This model is trained for simple synthetic grayscale shapes and is not intended for realistic scene generation.

**Attention Explorer**

Inspect saved baseline and attention-based GAN artifacts, including training curves, sample tensors, and attention-related results where exposed by the interface.

---

## 11. Model Artifacts and Experimental Results

The repository preserves selected model checkpoints and experiment outputs.

### Task 1 — LoRA artifacts

Location:

`outputs/task1/`

| Artifact | Description |
|---|---|
| `lora_custom_dataset/pytorch_lora_weights.safetensors` | Saved LoRA adapter |
| `training/training_loss.png` | Training-loss visualization |
| `baseline/` | Baseline image samples |
| `adapted/` | Adapted image samples |
| `comparison/` | Combined comparison visualization |

### Task 5 — Attention GAN artifacts

Location:

`models/cgan/task5/`

The following files are retained:

- Attention generator and discriminator weights.
- Baseline generator and discriminator weights.
- Model configuration.
- Architecture source copy.
- Training curves.
- Evaluation metrics.
- Validation report.
- Generated validation samples.

The corresponding portable experiment package is retained under:

`outputs/task5/colab_bundle/task5_artifacts/`

Additional local validation images are under:

`outputs/task5/validation/`

### Task 6 — Integrated GAN artifacts

Location:

`models/cgan/`

The repository includes the text-conditioned generator, discriminator, embeddings, and training-loss arrays.

The saved generator checkpoint was successfully loaded in the verification process and produced a finite 64 × 64 output in a forward-pass test.

### Attention comparisons

Location:

`outputs/comparisons/`

The folder contains:

- Baseline generator losses.
- Baseline discriminator losses.
- Attention generator losses.
- Attention discriminator losses.
- Baseline generated sample tensors.
- Attention-generated sample tensors.
- Attention weights.

### Dataset analysis

Location:

`outputs/graphs/`

The Flickr8k analysis CSV is retained as a summary of the dataset exploration experiment.

### Showcase images

Location:

`outputs/generated_images/`

The repository includes representative examples from the Realistic Studio and Internship Fine-Tuned application modes.

---

## 12. Dataset Information

### Flickr8k

Flickr8k is the public image-caption dataset used in the project's dataset analysis and selected image-caption workflows.

The configured Hugging Face dataset identifier is:

`intro/flickr8k`

The reusable loader is available at:

`src/data/flickr8k_loader.py`

The project uses streaming access where appropriate rather than requiring the full raw dataset to be stored in the Git repository.

### Dataset usage

Flickr8k is used for:

- Dataset exploration.
- Image-caption inspection.
- Caption preprocessing.
- Text-embedding experiments.
- The curated animal-focused subset used in the Task 1 fine-tuning workflow.

### Dataset storage

The repository includes dataset documentation under:

`data/README.md`

Raw and processed datasets are not committed as part of the source repository.

Where dataset access is shared through Google Drive, the folder should be made publicly accessible using the appropriate sharing permissions.

Dataset redistribution must follow the applicable dataset terms and license.

---

## 13. Reproducibility

The repository contains notebooks, scripts, configurations, checkpoints, and output artifacts to support project inspection and selected experiment reproduction.

### Experiment notebooks

| Notebook | Experiment |
|---|---|
| `00_environment_setup.ipynb` | Environment preparation |
| `01_dataset_analysis.ipynb` | Flickr8k dataset analysis |
| `02_text_preprocessing_embeddings.ipynb` | Text preprocessing and BERT |
| `03_conditional_gan.ipynb` | Conditional GAN |
| `04_attention_mechanism.ipynb` | Attention-based GAN |
| `05_pretrained_model_fine_tuning.ipynb` | Stable Diffusion LoRA |
| `06_integrated_text_to_image_pipeline.ipynb` | Integrated generation workflow |

### Task 1 reproduction

The reproduction script is:

```powershell
python scripts/reproduce_task1_lora.py
```

**Important:** This script may overwrite existing Task 1 experiment outputs. Preserve any results you need before rerunning it.

The configured workflow uses a curated subset of 60 examples and three training epochs. The full end-to-end training workflow has not been independently rerun as part of the current final smoke-verification pass.

### Task 5 reproduction and inspection

The Task 5 configuration, trained checkpoints, metrics, and validation results are stored under:

`models/cgan/task5/`

The canonical model architecture is:

`src/models/task5_attention_gan_model.py`

The Colab artifact bundle is preserved under:

`outputs/task5/colab_bundle/task5_artifacts/`

### General reproducibility considerations

Exact outputs can vary according to:

- Random seeds.
- Dataset version and access.
- Pretrained model revisions.
- Dependency versions.
- GPU hardware.
- Training configuration.
- Image preprocessing and generation settings.

A saved checkpoint and a successful loading test do not, by themselves, prove that the complete original training experiment can be reproduced.

---

## 14. Verification and Testing

The following technical smoke checks were completed on the development environment.

| Verification | Observed result |
|---|---|
| Python compilation | Passed for `app/`, `src/`, and `scripts/` |
| Application imports | 10 project modules imported successfully |
| Dependency consistency | `pip check` reported no broken requirements |
| Notebook structure | Seven notebooks parsed successfully |
| Task 1 LoRA checkpoint | 256 tensors loaded; all finite |
| Task 2 CGAN | Forward pass produced `(2, 1, 64, 64)` |
| Task 5 attention GAN | Four checkpoints loaded with strict state-dictionary matching |
| Task 5 forward passes | Baseline and attention models passed smoke tests |
| Task 6 generator | Checkpoint loaded and forward pass produced `(2, 1, 64, 64)` |
| Task 6 supporting artifacts | Discriminator, embeddings, and loss arrays were readable |
| Flickr8k summary | CSV loaded successfully |
| Git cleanliness | No tracked backup, Python cache, or virtual-environment files |

### Notebook execution metadata

The saved notebook execution counts observed during repository inspection were:

| Notebook | Executed code cells | Total code cells |
|---|---:|---:|
| Environment setup | 6 | 10 |
| Dataset analysis | 13 | 15 |
| Text preprocessing and embeddings | 2 | 16 |
| Conditional GAN | 20 | 20 |
| Attention mechanism | 29 | 29 |
| Pretrained-model fine-tuning | 0 | 17 |
| Integrated pipeline | 34 | 34 |

No saved error outputs were found during the notebook structure check.

These execution counts describe saved notebook metadata. They are not evidence that every notebook was freshly executed during the final verification.

The current verification is a technical smoke test, not a complete retraining or independent reproduction of all six experiments. Full application testing and end-to-end experiment reproduction should be considered separately.

---

## 15. Limitations and Considerations

### Stable Diffusion

The pretrained diffusion model requires appropriate computational resources and compatible dependencies. Image quality depends on prompt construction, model revision, generation settings, and hardware.

### LoRA fine-tuning

The custom fine-tuning experiment uses a relatively small curated image-caption subset. Its results should be interpreted as an experimental demonstration of parameter-efficient adaptation rather than evidence of broad domain generalization.

### Conditional GAN

The Conditional GAN generates low-resolution grayscale images. It is designed for basic synthetic shapes and does not target general-purpose realistic image synthesis.

### Text embeddings

The BERT representations provide contextual numerical features, but the quality of downstream image generation also depends on the generative model and its training procedure.

### Attention mechanisms

The attention experiment demonstrates cross-attention-based text conditioning. The saved metrics show successful synthetic-shape validation, but do not establish that the attention architecture outperforms the baseline.

### Application

The Streamlit interface connects selected project components. Its model-generation modes have different capabilities and limitations. A successful import or model checkpoint load does not guarantee that every user interaction has passed complete end-to-end testing.

### Hardware

GPU memory, CPU resources, available storage, and software compatibility affect model loading and generation.

### Licensing

Dataset and pretrained model terms must be reviewed before redistributing datasets, model weights, or other third-party materials.

---

## 16. Troubleshooting

### Streamlit command is not recognized

Activate the project virtual environment and use:

```powershell
python -m streamlit run .\run_app.py
```

### CUDA is unavailable

Verify the installed PyTorch build and NVIDIA driver compatibility:

```python
import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
```

### Model checkpoint cannot be found

Verify that the expected artifact exists in its documented directory. Pay particular attention to the Task 1 LoRA weights and the Task 5 and Task 6 GAN checkpoints.

### Pretrained model cannot be loaded

Check internet access, model identifiers, local Hugging Face cache contents, and dependency compatibility.

### Out-of-memory errors

Reduce the model's resource requirements where supported. Diffusion models and GAN experiments have different memory requirements.

### Generated results differ

Check the random seed, prompt, model revision, inference settings, and software versions before comparing outputs.

---

## 17. Internship Deliverable Coverage

| Internship task | Main implementation | Supporting evidence |
|---|---|---|
| Task 1 — Pretrained model fine-tuning | Stable Diffusion LoRA | Adapter weights, training plot, generated images |
| Task 2 — Conditional GAN | `src/models/conditional_gan.py` | Conditional GAN notebook and forward-pass verification |
| Task 3 — Text preprocessing and embeddings | `src/preprocessing/text_embeddings.py` | BERT preprocessing notebook |
| Task 4 — Public dataset analysis | `src/data/flickr8k_loader.py` | Dataset-analysis notebook and CSV |
| Task 5 — Attention mechanisms | `src/models/task5_attention_gan_model.py` | Baseline and attention checkpoints, metrics, validation report |
| Task 6 — Integrated pipeline | Text-conditioned GAN workflow | Integrated pipeline notebook and Task 6 checkpoints |

Supporting project deliverables also include:

- Streamlit application.
- Reusable model-service implementation.
- Experiment notebooks.
- Training and validation outputs.
- Project documentation.
- Final internship report.

This coverage table identifies the location of each implementation and its supporting evidence. It should not be interpreted as a claim that all original experiments have been independently rerun.

---

## 18. Conclusion

Text-to-Image Generation Studio brings together multiple approaches to text-conditioned image generation within a single internship project.

Through the six assigned tasks, the project explores:

- Pretrained diffusion-model adaptation.
- Conditional generative adversarial networks.
- Text preprocessing and contextual embeddings.
- Public image-caption dataset analysis.
- Cross-attention-based visual conditioning.
- An integrated text-to-image research pipeline.

The repository preserves reusable source code, model checkpoints, notebooks, evaluation artifacts, generated images, and a Streamlit application for demonstrating selected capabilities.

The project provides a foundation for further experimentation with text-conditioned generative models and future improvements to the application.

---

**Project:** Text-to-Image Generation Studio  
**Internship:** Data Science Internship — ElevanceSkills  
**Primary interface:** Streamlit  
**Deep-learning framework:** PyTorch  
**Generative approaches:** Stable Diffusion, LoRA, Conditional GAN, Attention-based GAN  
**Text encoder:** BERT  
**Dataset:** Flickr8k

