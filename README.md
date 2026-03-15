# ModelFineTuning

Fine-tuning experiment on the [`google-research-datasets/poem_sentiment`](https://huggingface.co/google-research-datasets/poem_sentiment) dataset.

## Dataset

| Split | Samples |
|---|---|
| Train | 892 |
| Validation | 105 |
| Test | 104 |

**Labels:** `0=negative`, `1=positive`, `2=no_impact`, `3=mixed`
Note: `mixed` class is absent from validation and test splits.

## Results

| Model | Test Accuracy |
|---|---|
| Naive Bayes (TF-IDF) | 66.35% |
| Naive Bayes (TF-IG) | 66.35% |
| Naive Bayes (BoW only) | 67.31% |
| Ministral-3-8B zero-shot | 54.81% |
| **Ministral-3-8B QLoRA fine-tuned** | **80.77%** |

## Key Observations

- **Naive Bayes** predicts almost exclusively `no_impact` (the majority class at 66%), which explains the deceptively high baseline accuracy.
- **Zero-shot Ministral-3-8B** scores below the Naive Bayes baseline (54.81%) — the base model is not instruction-tuned, so few-shot prompting alone is insufficient for this task.
- **QLoRA fine-tuning** (5 epochs, LoRA rank=16) pushes accuracy to **80.77%**, a +14pp improvement over the baseline and +26pp over zero-shot. Parse errors dropped to zero after fine-tuning.

## Setup

Model: [`unsloth/Ministral-3-8B-Base-2512-unsloth-bnb-4bit`](https://huggingface.co/unsloth/Ministral-3-8B-Base-2512-unsloth-bnb-4bit) (pre-quantized NF4)
Fine-tuning: QLoRA via PEFT + TRL (`r=16`, `lora_alpha=32`, `lr=2e-4`, 5 epochs)
Hardware: NVIDIA RTX 4090 (24GB VRAM)

## Notebooks

- [`naiveBayes.ipynb`](naiveBayes.ipynb) — Naive Bayes baselines
- [`mistral_finetune.ipynb`](mistral_finetune.ipynb) — Zero-shot evaluation + QLoRA fine-tuning
