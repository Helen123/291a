# CodeLlama 7B Fine-tuning with QLoRA (PPO & GRPO)

This repository contains code for fine-tuning CodeLlama 7B using QLoRA with PPO and GRPO reinforcement learning methods on the MBPP dataset. The implementation uses result-oriented reward functions based on code execution.

## Features

- **QLoRA Integration**: Memory-efficient 4-bit quantization with LoRA adapters
- **Multiple RL Methods**: Support for both PPO and GRPO training
- **Result-Oriented Rewards**: Reward function based on actual code execution and test passing
- **MBPP Dataset**: Training on the well-established MBPP programming benchmark
- **HuggingFace Integration**: Easy model sharing and deployment
- **Comprehensive Evaluation**: Built-in evaluation following bigcode-evaluation-harness standards

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with at least 16GB VRAM
- HuggingFace account (for model upload)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"  # Optional, for logging
export HF_ALLOW_CODE_EVAL="1"  # Required for code execution
```

## Quick Start

### Training with PPO

```bash
# Basic PPO training
bash run_training.sh --method ppo

# PPO with custom parameters
bash run_training.sh \
    --method ppo \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --push_to_hub \
    --hub_model_id "your-username/codellama-7b-mbpp-ppo"
```

### Training with GRPO

```bash
# Basic GRPO training
bash run_training.sh --method grpo

# GRPO with custom parameters
bash run_training.sh \
    --method grpo \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --push_to_hub
```

### Using Python Directly

```bash
# PPO training
python train.py \
    --method ppo \
    --model_name codellama/CodeLlama-7b-Python-hf \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --push_to_hub

# GRPO training
python train.py \
    --method grpo \
    --model_name codellama/CodeLlama-7b-Python-hf \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --num_epochs 3
```

## Configuration

### Model Configuration

The model uses QLoRA with the following default settings:

```python
# 4-bit quantization
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"

# LoRA configuration
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Training Configuration

Default training parameters:

```python
# Training settings
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 5e-5
num_train_epochs = 3
max_seq_length = 1024

# PPO specific
ppo_epochs = 4
mini_batch_size = 1
cliprange = 0.2
gamma = 1.0

# GRPO specific
grpo_alpha = 10.0
```

### Reward Function

The reward function is designed to be result-oriented:

```python
# Reward values
positive_reward = 1.0    # For passing all test cases
negative_reward = 0.0    # For failing test cases
syntax_penalty = -0.1    # For syntax errors
```

## Dataset and Evaluation

### MBPP Dataset

The training uses the MBPP (Mostly Basic Programming Problems) dataset:
- **Training**: 374 problems from the train split
- **Evaluation**: 500 problems from the test split
- **Format**: InCoder-style prompts with docstrings and example tests

### Evaluation Metrics

Evaluation follows the bigcode-evaluation-harness standard:
- **pass@k**: Percentage of problems solved with k attempts
- **Success Rate**: Percentage of problems with at least one correct solution
- **Code Execution**: Actual test case execution for verification

### Running Evaluation

```bash
# Evaluate a fine-tuned model
python evaluate.py ./checkpoints/best_model \
    --split test \
    --num_samples 15 \
    --temperature 0.1

# Evaluate with custom settings
python evaluate.py /path/to/model \
    --split test \
    --max_samples 100 \
    --temperature 0.2 \
    --num_samples 10
```

## Advanced Usage

### Custom Configuration

Create a custom configuration by modifying `config.py`:

```python
from config import ExperimentConfig

# Create custom config
config = ExperimentConfig()
config.model.lora_r = 128
config.training.learning_rate = 1e-4
config.training.num_train_epochs = 5
```

### Multi-GPU Training

The code automatically supports multi-GPU training with `device_map="auto"`. For explicit control:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python train.py --method ppo --batch_size 2
```

### Memory Optimization

For limited GPU memory:

```bash
# Reduce batch size and increase accumulation
python train.py \
    --method ppo \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 512
```

## Model Upload to HuggingFace Hub

### Automatic Upload

```bash
# Set your token
export HF_TOKEN="your_token_here"

# Train with auto-upload
bash run_training.sh \
    --method ppo \
    --push_to_hub \
    --hub_model_id "your-username/codellama-7b-mbpp-ppo"
```

### Manual Upload

```python
from model_utils import save_model_and_tokenizer

# Load your trained model
model, tokenizer = load_model_for_rl(config.model)

# Save and upload
save_model_and_tokenizer(
    model=model,
    tokenizer=tokenizer,
    save_path="./final_model",
    push_to_hub=True,
    hub_model_id="your-username/model-name",
    hub_token="your_token"
)
```

## Project Structure

```
finetuning/
├── config.py              # Configuration classes
├── data_utils.py           # MBPP data processing and reward functions
├── model_utils.py          # Model loading and QLoRA setup
├── ppo_trainer.py          # PPO training implementation
├── grpo_trainer.py         # GRPO training implementation
├── train.py               # Main training script
├── evaluate.py            # Evaluation script
├── run_training.sh        # Training shell script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Results and Benchmarks

Expected performance improvements over base CodeLlama:

| Method | pass@1 | pass@5 | pass@10 |
|--------|--------|--------|---------|
| Base CodeLlama 7B | ~30% | ~45% | ~55% |
| + PPO Fine-tuning | ~35-40% | ~50-55% | ~60-65% |
| + GRPO Fine-tuning | ~33-38% | ~48-53% | ~58-63% |

*Note: Results may vary based on training configuration and hardware.*

## Monitoring and Logging

### Weights & Biases

The training automatically logs to W&B:

```bash
# Enable W&B logging (default)
python train.py --method ppo

# Disable W&B logging
python train.py --method ppo --no_wandb
```

### Local Logging

All training runs are logged to `./logs/`:
- Training commands: `training_commands.log`
- Training output: `training_ppo_YYYYMMDD_HHMMSS.log`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   python train.py --batch_size 1 --gradient_accumulation_steps 16
   ```

2. **Code Execution Timeout**:
   ```python
   # Increase timeout in config
   config.training.code_execution_timeout = 15.0
   ```

3. **Model Upload Failure**:
   ```bash
   # Check HF token
   huggingface-cli login
   ```

### Performance Tips

1. **Use mixed precision**: Enabled by default with `torch.float16`
2. **Optimize batch size**: Balance between memory and training stability
3. **Monitor GPU utilization**: Use `nvidia-smi` to check resource usage
4. **Use gradient checkpointing**: For very large models

## Citation

If you use this code, please cite:

```bibtex
@misc{codellama-mbpp-finetuning,
  title={CodeLlama Fine-tuning with QLoRA on MBPP},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/codellama-mbpp-finetuning}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [CodeLlama](https://github.com/facebookresearch/codellama) by Meta
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) by Google Research
- [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) by BigCode
- [TRL](https://github.com/huggingface/trl) by HuggingFace
- [PEFT](https://github.com/huggingface/peft) by HuggingFace 