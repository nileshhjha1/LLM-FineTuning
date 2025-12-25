# Fine-Tuning with Unsloth: Llama-3.2-3B-Instruct Tutorial

##  Overview
This notebook demonstrates how to fine-tune the **Llama-3.2-3B-Instruct** model using **Unsloth**, optimizing it for enhanced "thinking" capabilities similar to DeepSeek-R1. The training leverages the **ServiceNow-AI/R1-Distill-SFT** dataset.

---

##  Quick Start

### 1. Install Unsloth
```bash
pip install -q unsloth
# Install latest nightly Unsloth
pip install -q --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

### 2. Load the Quantized Model
```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None  # auto detection
load_in_4bit = True  # reduces memory usage

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

### 3. Prepare Model for QLoRA Fine-Tuning
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                           # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,                  # weight of LoRA activations
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

---

## 📖 Key Concepts Explained

### **Model Loading Parameters**
- `model_name`: Pre-trained model to load
- `max_seq_length`: Maximum token sequence length (2048 tokens)
- `dtype`: Data type for computations (auto-detected based on GPU)
- `load_in_4bit`: Enables 4-bit quantization for memory efficiency

### **LoRA Configuration Parameters**
- `r`: Rank of low-rank matrices (higher = more parameters)
- `target_modules`: Model components where LoRA adapters are inserted
- `lora_alpha`: Scaling factor for LoRA updates
- `lora_dropout`: Dropout rate to prevent overfitting
- `use_gradient_checkpointing`: Reduces memory during training
- `random_state`: Ensures reproducibility

---

## 🎯 Purpose
- Fine-tune Llama-3.2-3B-Instruct to mimic DeepSeek-R1 reasoning capabilities
- Use efficient QLoRA (4-bit quantization + LoRA) for memory-efficient training
- Leverage the ServiceNow-AI/R1-Distill-SFT dataset for specialized training

---



## ✅ Notes
- Uses **Unsloth** for 2x faster fine-tuning
- Supports **RoPE (Rotary Positional Embedding)** scaling
- Compatible with Tesla T4/V100 (float16) and Ampere+ GPUs (bfloat16)

---
