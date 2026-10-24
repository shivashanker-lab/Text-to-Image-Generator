# Text-to-Image Generation Studio

**Data Science Internship Project — ElevanceSkills**

A text-to-image generation project exploring pretrained diffusion models, Low-Rank Adaptation (LoRA), text preprocessing, BERT embeddings, Conditional Generative Adversarial Networks (Conditional GANs), attention mechanisms, and dataset analysis.

The project brings these components together through a Streamlit application that provides a unified interface for image generation and research demonstrations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Objectives](#2-project-objectives)
3. [Application Features](#3-application-features)
4. [Internship Tasks](#4-internship-tasks)
5. [System Architecture](#5-system-architecture)
6. [Repository Structure](#6-repository-structure)
7. [Technology Stack](#7-technology-stack)
8. [Environment and Requirements](#8-environment-and-requirements)
9. [Running the Application](#9-running-the-application)
10. [Using the Application](#10-using-the-application)
11. [Model Artifacts and Experiment Outputs](#11-model-artifacts-and-experiment-outputs)
12. [Dataset Handling](#12-dataset-handling)
13. [Reproducibility](#13-reproducibility)
14. [Limitations and Considerations](#14-limitations-and-considerations)
15. [Troubleshooting](#15-troubleshooting)
16. [Internship Deliverable Coverage](#16-internship-deliverable-coverage)

---

## 1. Project Overview

Text-to-Image Generation Studio is an educational and research-oriented project developed as part of the Data Science Internship at ElevanceSkills.

The project explores how natural-language descriptions can be processed and used by generative models to produce images. It covers multiple stages of a text-to-image workflow, from dataset analysis and text representation to model experimentation, fine-tuning, saved artifacts, and application integration.

The implementation includes three distinct image-generation approaches:

- **Stable Diffusion:** Uses a pretrained diffusion pipeline for general-purpose image generation.
- **Stable Diffusion with LoRA:** Demonstrates parameter-efficient adaptation using saved internship fine-tuning weights.
- **BERT with Conditional GAN:** Demonstrates an experimental text-conditioned generation pipeline that produces low-resolution grayscale images.

An additional Attention Explorer interface presents saved results from baseline and attention-based GAN experiments.

These approaches serve different purposes. The Stable Diffusion pipelines target general image generation, while the Conditional GAN provides a separate experimental implementation of text-conditioned generation.

### What this project demonstrates

- Exploring a public image-caption dataset.
- Preprocessing and representing natural-language text.
- Working with BERT text embeddings.
- Experimenting with Conditional GAN architectures.
- Investigating attention mechanisms in generative models.
- Adapting a pretrained diffusion model with LoRA.
- Organizing checkpoints, generated images, and experiment results.
- Integrating machine-learning components into a Streamlit application.

---

## 2. Project Objectives

The main objectives of this project are to:

1. Understand the relationship between natural-language descriptions and image generation.
2. Explore image-caption datasets and summarize their characteristics.
3. Transform text into numerical representations suitable for machine-learning models.
4. Experiment with text-conditioned generative architectures.
5. Investigate differences between baseline and attention-based GAN experiments.
6. Explore parameter-efficient fine-tuning of a pretrained diffusion model.
7. Build a unified application for demonstrating the implemented models.
8. Maintain an organized structure for source code, notebooks, model artifacts, and outputs.
9. Document experimental results, limitations, and reproduction considerations.

---

## 3. Application Features

The project includes a Streamlit application with the following modes and supporting interfaces.

| Application mode | Technology | Purpose |
|---|---|---|
| Realistic Studio | Stable Diffusion | General-purpose text-to-image generation. |
| Internship Fine-Tuned | Stable Diffusion with LoRA | Demonstration of the internship fine-tuning artifact. |
| Research Lab | BERT with Conditional GAN | Experimental text-conditioned generation. |
| Attention Explorer | Saved GAN experiment results | Inspection of baseline and attention-based experiment artifacts. |
| Project Information | Project documentation and status | Overview of the application and its components. |

### Core application capabilities

The interface includes features such as:

- Text-prompt input.
- Image generation through supported model pipelines.
- Seed configuration.
- Inference-step controls where supported.
- Guidance-scale controls where supported.
- Negative-prompt controls for Stable Diffusion modes.
- Generation details and device information.
- Generation-time reporting.
- PNG image download.
- Preview of the generated image.
- Access to saved attention-experiment results.
- Model and artifact status information.

The available controls and resource requirements depend on the selected generation pipeline.

---

## 4. Internship Tasks

The technical work is organized into six internship tasks.

### Task 1 — Pretrained Model and Custom Dataset

This task explores adapting a pretrained Stable Diffusion model using Low-Rank Adaptation (LoRA).

The workflow includes preparing image-caption data, configuring a pretrained diffusion pipeline, applying parameter-efficient adaptation, saving the adapter weights, and comparing baseline and adapted image outputs.

**Main artifact:**

`outputs/task1/lora_custom_dataset/pytorch_lora_weights.safetensors`

**Additional outputs:**

- Three baseline example images.
- Three adapted example images.
- A consolidated baseline-versus-adapted comparison image.
- A training-loss plot.

The corresponding files are organized under `outputs/task1/`.

The saved outputs provide examples of the baseline and adapted pipelines. Pixel differences between images do not, by themselves, establish that fine-tuning improved image quality. Comparisons should account for prompt differences and other generation settings.

### Task 2 — Conditional Generative Adversarial Network

This task explores a Conditional GAN for image generation conditioned on text representations.

The generator uses a latent input and text-conditioned information to produce an image. The discriminator checkpoint is retained with the other training artifacts.

The current research model produces **64 × 64 grayscale images**.

**Main artifacts:**

- `models/cgan/task6_text_conditioned_generator.pth`
- `models/cgan/task6_text_conditioned_discriminator.pth`
- `models/cgan/task6_text_embeddings.pt`
- `models/cgan/task6_generator_losses.npy`
- `models/cgan/task6_discriminator_losses.npy`

The Conditional GAN is an experimental model and should not be expected to match the general-purpose image quality of a large pretrained diffusion model.

### Task 3 — Text Preprocessing and BERT Embeddings

This task transforms text descriptions into numerical representations that can be used by machine-learning models.

The workflow includes:

1. Cleaning text.
2. Tokenizing descriptions.
3. Generating contextual text representations using BERT.
4. Producing 768-dimensional embeddings.
5. Supplying text representations to the research pipeline.

The implementation is organized within the project's `src/` modules and associated notebooks.

### Task 4 — Public Dataset Analysis

This task analyzes an image-caption dataset to understand its structure and text characteristics.

The analysis covers image counts, caption statistics, and other dataset-summary information.

A saved summary is available at:

`outputs/graphs/flickr8k_analysis_results.csv`

The saved analysis records include the following reported statistics:

| Metric | Recorded result |
|---|---:|
| Training images | 6,000 |
| Validation images | 1,000 |
| Test images | 1,000 |
| Total images | 8,000 |
| Captions per image | 5 |
| Training captions analyzed | 30,000 |
| Average caption length | Approximately 11.78 words |
| Unique words reported | 7,705 |

These values describe the saved analysis output. Their exact interpretation depends on the dataset splits and counting procedures used by the corresponding analysis code.

### Task 5 — Attention Mechanisms

This task explores attention-related experiments by comparing saved baseline GAN and attention-based GAN results.

The saved artifacts include:

- Baseline generator-loss values.
- Attention-based generator-loss values.
- Baseline discriminator-loss values.
- Attention-based discriminator-loss values.
- Generated sample tensors.
- Attention-weight tensors.

The artifacts are stored in `outputs/comparisons/`.

The Attention Explorer provides an interface for inspecting these existing results. Their presence supports experiment inspection, but does not establish that the attention-based model outperforms the baseline without an appropriate evaluation.

### Task 6 — Integrated Text-to-Image Pipeline

This task connects text processing, BERT embeddings, and the Conditional GAN into a text-conditioned generation workflow.

The conceptual pipeline is:

```text
Natural-Language Prompt
          |
          v
    Text Preprocessing
          |
          v
   BERT Tokenization
          |
          v
  768-D Text Embedding
          |
          v
 Text-Conditioned GAN
          |
          v
 64 × 64 Grayscale Image
```

The Research Lab mode provides access to this implementation through the Streamlit application.

The project retains the relevant model checkpoints and experiment artifacts separately from the application code.

---

## 5. System Architecture

The project uses a modular structure that separates the application interface, model-serving logic, reusable source code, notebooks, model artifacts, and experiment outputs.

```text
                  User Prompt
                      |
                      v
            Streamlit Application
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
      Realistic   Internship   Research
       Studio     Fine-Tuned     Lab
          |           |           |
          v           v           v
       Stable      Stable       BERT +
      Diffusion   Diffusion      CGAN
                      |
                      v
                 LoRA Adapter

          Supporting Interfaces
                      |
          +-----------+-----------+
          |                       |
          v                       v
   Attention Explorer      Project Information
          |
          v
 Saved Experiment Artifacts
```

### Architectural responsibilities

**Application layer — `app/`**

Contains the Streamlit interface, configuration, and model-service components.

**Reusable source code — `src/`**

Contains shared project modules for the underlying text-processing and machine-learning workflows.

**Notebooks — `notebooks/`**

Preserve the step-by-step internship experiments.

**Model artifacts — `models/`**

Stores the selected Conditional GAN checkpoints, embeddings, and related training-loss arrays.

**Experiment outputs — `outputs/`**

Stores generated images, analysis results, fine-tuning outputs, and attention-experiment artifacts.

**Entry point — `run_app.py`**

Provides the root-level application launch entry point.

---

## 6. Repository Structure

The main project structure is organized as follows:

```text
Text-to-Image-Generator/
|
|-- app/
|   |-- __init__.py
|   |-- app.py
|   |-- config.py
|   `-- services/
|       |-- __init__.py
|       `-- model_service.py
|
|-- data/
|   `-- README.md
|
|-- models/
|   |-- README.md
|   `-- cgan/
|       |-- task6_text_conditioned_generator.pth
|       |-- task6_text_conditioned_discriminator.pth
|       |-- task6_text_embeddings.pt
|       |-- task6_generator_losses.npy
|       `-- task6_discriminator_losses.npy
|
|-- notebooks/
|   |-- 00_environment_setup.ipynb
|   |-- 01_dataset_analysis.ipynb
|   |-- 02_text_preprocessing_embeddings.ipynb
|   |-- 03_conditional_gan.ipynb
|   |-- 04_attention_mechanism.ipynb
|   |-- 05_pretrained_model_fine_tuning.ipynb
|   `-- 06_integrated_text_to_image_pipeline.ipynb
|
|-- outputs/
|   |-- comparisons/
|   |-- generated_images/
|   |-- graphs/
|   `-- task1/
|       |-- baseline/
|       |-- adapted/
|       |-- comparison/
|       |-- training/
|       `-- lora_custom_dataset/
|
|-- reports/
|
|-- scripts/
|   |-- README.md
|   `-- reproduce_task1_lora.py
|
|-- src/
|
|-- .gitignore
|-- README.md
`-- run_app.py
```

Some directories contain additional files that are not expanded in this overview. The actual repository contents should be treated as authoritative if the structure changes.

---

## 7. Technology Stack

| Technology | Role in the project |
|---|---|
| Python | Main programming language. |
| PyTorch | Tensor operations and deep-learning models. |
| Streamlit | Interactive application interface. |
| Hugging Face Diffusers | Diffusion-model pipeline. |
| Hugging Face Transformers | Tokenization and transformer-based text representations. |
| PEFT / LoRA | Parameter-efficient model adaptation. |
| Stable Diffusion | Pretrained text-to-image generation. |
| BERT | Text encoding and contextual embeddings. |
| Conditional GAN | Experimental text-conditioned image generation. |
| NumPy | Numerical arrays and experiment data. |
| Pandas | Dataset statistics and analysis tables. |
| Pillow | Image loading and manipulation. |
| Jupyter Notebook | Experiment development and documentation. |

---

## 8. Environment and Requirements

### Development environment

The development environment recorded for this project includes:

| Component | Recorded environment |
|---|---|
| Operating system | Windows |
| Python | 3.12.10 |
| PyTorch | 2.9.1+cu130 |
| Diffusers | 0.40.0 |
| Streamlit | 1.64.0 |
| GPU | NVIDIA RTX 3050 with 6 GB VRAM |

These versions describe the development environment and are not a guarantee that every combination of hardware and dependencies will behave identically.

### Requirements for running the application

The application requires a compatible Python environment and the relevant project dependencies, including the libraries used by the selected model pipeline.

Stable Diffusion inference and model loading can require substantial GPU memory, storage, and model-download time. Requirements vary by pipeline, configuration, and hardware.

Pretrained model weights are not all bundled with the repository. Depending on the configuration and local cache, the application may need internet access to obtain them.

Before running the project in a fresh environment, confirm the available dependency instructions and install compatible versions of PyTorch, CUDA-related components where applicable, Diffusers, Transformers, Streamlit, and the other required packages. Do not assume that the development environment's installed packages are automatically present on another machine.

---

## 9. Running the Application

Run the following commands from the project root in Windows PowerShell.

### Step 1 — Activate the virtual environment

If the project's virtual environment already exists:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2 — Launch Streamlit

```powershell
python -m streamlit run .\run_app.py
```

Alternatively, explicitly use the project's virtual-environment interpreter:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run ".\run_app.py"
```

### Step 3 — Open the application

Open the local URL printed by Streamlit in the terminal.

Keep the terminal running while using the application. To stop the application, return to the terminal and press `Ctrl+C`.

**Note:** Successful launch requires the appropriate dependencies, configuration, and any model artifacts required by the selected mode. A successful Streamlit launch does not, by itself, confirm that every model pipeline has loaded successfully.

---

## 10. Using the Application

### Realistic Studio

1. Select Realistic Studio.
2. Enter a descriptive image prompt.
3. Configure the available generation settings.
4. Start generation.
5. Inspect and download the resulting image.

This mode uses the configured pretrained Stable Diffusion pipeline.

### Internship Fine-Tuned

1. Select Internship Fine-Tuned.
2. Enter a prompt.
3. Configure the available generation settings.
4. Start generation.
5. Inspect and download the result.

This mode uses Stable Diffusion with the internship LoRA adapter. Its operation depends on the configured base model and the availability of the saved adapter weights.

### Research Lab

1. Select Research Lab.
2. Enter a text description.
3. Start generation.
4. Inspect the experimental output.

This mode demonstrates the BERT-conditioned Conditional GAN, which produces 64 × 64 grayscale images. It is most appropriate for inspecting the behavior of the experimental model rather than expecting general-purpose photorealistic images.

### Attention Explorer

Use Attention Explorer to inspect the available baseline and attention-based GAN artifacts. These saved outputs provide a view of the experiments without requiring every result to be regenerated during the application session.

---

## 11. Model Artifacts and Experiment Outputs

The repository keeps key research artifacts separate from the application code.

### Conditional GAN artifacts

Location: `models/cgan/`

| Artifact | Purpose |
|---|---|
| `task6_text_conditioned_generator.pth` | Generator checkpoint. |
| `task6_text_conditioned_discriminator.pth` | Discriminator checkpoint. |
| `task6_text_embeddings.pt` | Saved text embeddings used by the research pipeline. |
| `task6_generator_losses.npy` | Generator training-loss values. |
| `task6_discriminator_losses.npy` | Discriminator training-loss values. |

### Attention-experiment artifacts

Location: `outputs/comparisons/`

The directory contains saved baseline and attention-based GAN losses, generated sample tensors, and attention weights.

These artifacts can be used to inspect recorded experiment outputs and to support the corresponding research documentation.

### Dataset-analysis outputs

Location: `outputs/graphs/`

The dataset-analysis summary is stored as:

`flickr8k_analysis_results.csv`

### Generated showcase images

Location: `outputs/generated_images/`

The repository includes:

- `realistic_studio.png`
- `internship_finetuned.png`

These images serve as examples of the corresponding application modes.

### Task 1 outputs

Location: `outputs/task1/`

| Path | Contents |
|---|---|
| `baseline/` | Baseline image examples. |
| `adapted/` | Adapted image examples. |
| `comparison/` | Consolidated comparison image. |
| `training/` | Training-loss plot. |
| `lora_custom_dataset/` | Saved LoRA adapter weights. |

Generated images, numerical arrays, and model checkpoints should be interpreted according to the experiment that produced them. Their existence alone does not establish model superiority or guarantee that a training run can be reproduced exactly.

---

## 12. Dataset Handling

The project explores a public image-caption dataset, including Flickr8k analysis and text-conditioned generation experiments.

Dataset files are intentionally kept outside Git where appropriate. The `data/` directory contains documentation, while raw and processed data directories may remain empty when data is obtained through the configured dataset-loading workflow.

This arrangement avoids committing a complete dataset unnecessarily and keeps the repository focused on code, documentation, model artifacts, and selected experiment results.

### Dataset handling considerations

- Follow the dataset source's applicable terms and license.
- Do not assume that a publicly accessible dataset can be redistributed without restrictions.
- Keep large raw datasets and temporary processing files out of Git unless explicitly required.
- Confirm the configured dataset identifier and download procedure before attempting a fresh reproduction.
- Do not include private credentials or access tokens in notebooks, scripts, or documentation.

The dataset source and configuration used by an individual experiment should be checked in its corresponding notebook or script.

---

## 13. Reproducibility

The project preserves notebooks, scripts, checkpoints, and selected outputs to document the implementation and experiments.

### Reproduction resources

- The numbered notebooks document the research and development workflow.
- `scripts/reproduce_task1_lora.py` provides a Task 1 training/reproduction path.
- `models/cgan/` contains selected Conditional GAN artifacts.
- `outputs/` contains saved experiment outputs and generated images.
- `app/` contains the integrated application implementation.

### Important reproducibility notes

The Task 1 reproduction script is configured for a small demonstration run using 60 image-caption pairs and three training epochs. Its complete end-to-end training execution has not been independently revalidated in the current release review.

Running this script can overwrite files under `outputs/task1/`. Back up any outputs you want to retain before starting a training run.

Exact results can vary with dependency versions, random seeds, dataset access, model versions, hardware, and generation settings. Reproduction should be established through an actual run and verification of the outputs rather than inferred from the presence of a script or saved checkpoint.

For reliable comparisons, hold the prompt, seed, model configuration, and other relevant generation settings constant.

---

## 14. Limitations and Considerations

### Stable Diffusion

Image quality and prompt adherence depend on the pretrained model, prompt design, generation settings, and available computational resources. Results can vary across runs and environments.

### Internship LoRA

The internship adaptation uses a relatively small dataset and training configuration. Its behavior can differ from the base model, and fine-tuning should not be assumed to improve every prompt or image-quality measure.

A fair comparison requires equivalent prompts and controlled generation settings.

### Conditional GAN

The research GAN generates 64 × 64 grayscale images. Its capabilities are more limited than those of large pretrained diffusion models, especially for complex scenes, realistic textures, multiple objects, and detailed compositions.

### Attention experiments

Saved losses, image samples, and attention weights provide material for inspection. They do not establish an improvement in image quality without a suitable evaluation method, controlled comparisons, and reproducible measurements.

### Hardware and dependencies

The application depends on compatible packages and available computational resources. Model loading and image generation may fail when memory, dependencies, cached files, or model downloads are insufficient.

### Dataset and model licensing

The license and usage conditions of the dataset, pretrained model, and any redistributed checkpoints must be respected. Review the relevant source documentation before redistribution or public release.

---

## 15. Troubleshooting

### Streamlit command is not recognized

Activate the virtual environment and use:

```powershell
python -m streamlit run .\run_app.py
```

If the command fails because Streamlit is unavailable, verify that it is installed in the same Python environment being used to launch the application.

### Application cannot find a model artifact

Verify the expected file path, filename, and configuration. In particular, check the LoRA weights under `outputs/task1/lora_custom_dataset/` and the Conditional GAN files under `models/cgan/`.

### Model download or loading fails

Check network availability, model identifiers, local cache availability, dependency compatibility, and any required access permissions.

### CUDA or GPU memory errors occur

Check PyTorch's device configuration and available GPU memory. Try a supported lower-resource configuration where possible. Avoid assuming that successful model loading guarantees that all image-generation settings will fit in memory.

### Generated images differ between runs

Check the random seed, prompt, selected model, inference settings, model revision, and dependency versions. Differences are not necessarily evidence of a problem when relevant settings or software versions differ.

---

## 16. Internship Deliverable Coverage

| Internship task | Implementation or evidence |
|---|---|
| Pretrained model and custom dataset | Stable Diffusion LoRA workflow and Task 1 artifacts. |
| Conditional GAN | Generator and discriminator checkpoints under `models/cgan/`. |
| Text preprocessing and embeddings | Reusable preprocessing code and BERT-related notebooks. |
| Public dataset analysis | Dataset-analysis notebook and saved CSV results. |
| Attention mechanisms | Saved baseline and attention-based GAN experiment artifacts. |
| Integrated text-to-image pipeline | BERT-conditioned Conditional GAN research workflow. |
| Final application | Streamlit application launched through `run_app.py`. |
| Experiment documentation | Notebooks, README files, saved outputs, and reports. |

This table summarizes the project's intended task coverage. The presence of an artifact should not be treated as proof that a feature has passed every test; verification results should be documented separately.

---

## Project Summary

Text-to-Image Generation Studio brings together several approaches to text-conditioned image generation within one internship project.

It combines dataset exploration, text preprocessing, BERT embeddings, Conditional GAN experiments, attention mechanisms, pretrained diffusion models, LoRA adaptation, and a Streamlit-based interface.

The project preserves research artifacts and example outputs while keeping the application and reusable source code organized for inspection and further development.

**Project:** Text-to-Image Generation Studio

**Internship:** Data Science Internship — ElevanceSkills

**Primary interface:** Streamlit
**Main research components:** Stable Diffusion, LoRA, BERT, Conditional GANs, and attention experiments
