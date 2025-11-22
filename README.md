# Text-SQL-Fine-Tune
Fine-tunes the Llama 3.2 1B Instruct model on the Spider text-to-SQL dataset



## Overview

The code implements a complete fine-tuning pipeline that:
- Loads a pre-trained Llama 3.2 1B Instruct model
- Applies QLoRA (Quantized Low-Rank Adaptation) for efficient training
- Fine-tunes on the Spider dataset (text-to-SQL task)
- Uses Unsloth for optimized training speed and memory efficiency

## Requirements

### Installation

```bash
# Install required packages
pip install unsloth
pip install transformers
pip install datasets
pip install trl
pip install torch
pip install accelerate
```
## Hardware Requirements

  GPU with at least 4GB VRAM (for 1B model)

  CUDA support


#  Important Optimization Parameters

## Memory Optimization Parameters

| Parameter | Value | Purpose | Optimization Impact |
|-----------|-------|---------|-------------------|
| `load_in_4bit=True` | Boolean | 4-bit quantization | **Reduces VRAM usage by ~75%** |
| `max_seq_length=1024` | Integer | Sequence length limit | Controls memory usage during training |
| `per_device_train_batch_size=1` | Integer | Batch size per GPU | **Critical for fitting in VRAM** |
| `gradient_accumulation_steps=8` | Integer | Effective batch size | Enables larger effective batches with low memory |
| `use_gradient_checkpointing="unsloth"` | String | Memory for compute trade-off | **Reduces memory by ~30%** at cost of speed |
| `dataloader_pin_memory=False` | Boolean | CPU memory pinning | Reduces CPU memory usage |

## Training Efficiency Parameters

| Parameter | Value | Purpose | Optimization Impact |
|-----------|-------|---------|-------------------|
| `r=16` | Integer | LoRA rank | **Higher = more parameters, better quality** |
| `lora_alpha=16` | Integer | LoRA scaling | Affects learning rate scaling for LoRA layers |
| `fp16=not torch.cuda.is_bf16_supported()` | Auto-detected | FP16 precision | **2-3x speedup** on compatible GPUs |
| `bf16=torch.cuda.is_bf16_supported()` | Auto-detected | BF16 precision | Better numerical stability than FP16 |
| `optim="adamw_8bit"` | String | 8-bit optimizer | **Reduces optimizer memory by ~50%** |
| `packing=False` | Boolean | Sequence packing | Can improve throughput but may hurt convergence |

## Performance & Quality Parameters

| Parameter | Value | Purpose | Optimization Impact |
|-----------|-------|---------|-------------------|
| `learning_rate=2e-4` | Float | Learning rate | **Critical for convergence** - too high = unstable, too low = slow |
| `max_steps=100` | Integer | Training duration | Balances training time vs. performance |
| `warmup_steps=10` | Integer | Learning rate warmup | **Prevents early training instability** |
| `lora_dropout=0` | Float | Dropout for LoRA | Regularization - 0 for small datasets |
| `weight_decay=0.01` | Float | L2 regularization | Prevents overfitting |
| `lr_scheduler_type="linear"` | String | Learning rate schedule | Controls how learning rate decreases |

## Architecture Selection Parameters

### Target Modules Selection
```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",      # Attention layers
    "gate_proj", "up_proj", "down_proj",         # MLP layers
]
