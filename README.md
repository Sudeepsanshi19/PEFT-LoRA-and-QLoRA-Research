# PEFT LoRA and QLoRA Research

This project is a research-style Google Colab notebook that compares parameter-efficient fine-tuning methods for large language models on IMDb sentiment classification.

The notebook evaluates three approaches using `Qwen/Qwen2.5-1.5B-Instruct` as the primary backbone, with `TinyLlama/TinyLlama-1.1B-Chat-v1.0` available as a fallback if GPU memory is insufficient:

- Zero-shot inference with the base model
- LoRA fine-tuning in 16-bit precision
- QLoRA fine-tuning with 4-bit NF4 quantization

The experiment tracks model quality, memory use, training time, trainable parameters, model size, confusion matrices, and visual comparison charts.

## Notebook

Main notebook:

```text
PEFT_LoRA_QLoRA_Research.ipynb
```

The notebook is designed for Google Colab and expects a CUDA GPU runtime.

## Research Objective

Full fine-tuning large language models requires updating billions of parameters, which is expensive in GPU memory and compute. This notebook studies whether PEFT methods can reduce trainable parameter count and memory use while maintaining or improving performance.

The task is binary sentiment classification on the IMDb dataset:

- `Negative`
- `Positive`

Each movie review is converted into an instruction-style prompt:

```text
### Instruction:
Determine whether the movie review is Positive or Negative.

### Review:
<review text>

### Response:
<label>
```

## Dataset

The notebook uses the IMDb dataset from Hugging Face Datasets.

Configuration:

| Setting | Value |
| --- | ---: |
| Training samples | 10,000 |
| Test samples | 2,000 |
| Maximum sequence length | 512 |
| Seed | 42 |
| Labels | Negative, Positive |

Long reviews are pre-truncated before prompt formatting.

## Methods

### Zero-Shot

The base model is evaluated directly without fine-tuning. It generates either `Positive` or `Negative` from the instruction prompt.

### LoRA

LoRA adds trainable low-rank adapter weights to selected attention projection layers while keeping the base model frozen.

LoRA configuration:

| Setting | Value |
| --- | ---: |
| Rank `r` | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Trainable parameters | 2.1791M |
| Percent trainable | 0.141% |

In the recorded run, LoRA training hit an out-of-memory error on the T4 runtime, so its final metrics matched the zero-shot baseline.

### QLoRA

QLoRA combines LoRA adapters with 4-bit quantization to reduce memory use during fine-tuning.

QLoRA configuration:

| Setting | Value |
| --- | --- |
| Quantization | 4-bit NF4 |
| Double quantization | Enabled |
| Optimizer | `paged_adamw_8bit` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |

## Training Configuration

| Setting | Value |
| --- | ---: |
| Epochs | 1 |
| Per-device batch size | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 |
| Learning rate | 2e-4 |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Precision | FP16 or BF16 depending on GPU support |

## Results

Recorded results from the notebook run:

| Method | Accuracy | Precision | Recall | F1 Score | Train Time | Peak GPU | Trainable Params | Model Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-Shot | 0.9320 | 0.9106 | 0.9580 | 0.9337 | N/A | 7089 MB | 0 | 2944 MB |
| LoRA | 0.9320 | 0.9106 | 0.9580 | 0.9337 | 0.2 min | 7098 MB | 2.1791M | 2953 MB |
| QLoRA | 0.9470 | 0.9686 | 0.9240 | 0.9458 | 209.2 min | 4646 MB | 2.1791M | 1524 MB |

QLoRA achieved the best accuracy and F1 score while using substantially less peak GPU memory than the full-precision zero-shot and LoRA runs.

## Key Findings

- Zero-shot performance was already strong on IMDb sentiment classification.
- The LoRA run encountered an OOM during training on the T4 GPU, so it did not improve over zero-shot in the captured output.
- QLoRA completed training successfully and improved accuracy from `0.9320` to `0.9470`.
- QLoRA reduced peak GPU memory usage to about `4.6 GB`, compared with about `7.1 GB` for zero-shot and LoRA evaluation.
- Both LoRA and QLoRA trained only about `2.18M` adapter parameters instead of updating the full model.


## Requirements

The notebook installs the following main dependencies:

```text
transformers==4.44.2
peft==0.12.0
trl==0.10.1
bitsandbytes==0.49.2
accelerate==0.34.2
datasets==2.21.0
scikit-learn
matplotlib
seaborn
sentencepiece
protobuf
tabulate
```

Hardware used in the notebook:

```text
Google Colab Tesla T4 GPU, 16 GB VRAM
```

Estimated runtime:

```text
3-4 hours on a T4 GPU
```

If the primary Qwen model causes an out-of-memory error, the notebook can fall back to TinyLlama.

## Project Structure

```text
.
+-- PEFT_LoRA_QLoRA_Research.ipynb
+-- README.md
```

## Notes

- The notebook requires internet access to download models and the IMDb dataset from Hugging Face.
- GPU memory is the main constraint. QLoRA is the most practical path for a 16 GB T4 runtime.
- The LoRA results should be interpreted cautiously because the recorded LoRA training run did not complete successfully.
- Re-running the notebook may produce slightly different timing and memory numbers depending on Colab hardware, package versions, and runtime state.
