# Enhancing LLM Code Generation via Reinforcement Learning Techniques

This repository contains code for fine-tuning CodeLlama and other code generation models using QLoRA with PPO and GRPO reinforcement learning methods on MBPP and HumanEval datasets. The implementation uses result-oriented reward functions based on actual code execution.

## Features

- **🚀 Unified Training Script**: Single `train.py` script supports multiple algorithms and models
- **🔧 QLoRA Integration**: Memory-efficient 4-bit quantization with LoRA adapters  
- **🤖 Multiple RL Methods**: Support for both PPO and GRPO training
- **🎯 Result-Oriented Rewards**: Reward functions based on actual code execution and test passing
- **📊 Multi-Dataset Support**: Training on MBPP and HumanEval datasets
- **🔗 Auto-Merging**: Automatic adapter merging after training
- **🔬 Auto-Evaluation**: Built-in evaluation following bigcode-evaluation-harness standards
- **📈 Comprehensive Logging**: WandB integration and detailed metrics

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with at least 16GB VRAM
- HuggingFace account (for model upload)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd codegeneration
```

2. Install dependencies:
```bash
cd finetuning
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"  # Optional, for logging
export HF_ALLOW_CODE_EVAL="1"  # Required for code execution
```

## Quick Start

### 🚀 Unified Training Script

The main training interface is through `finetuning/train.py`:

```bash
cd finetuning

# Basic usage: method GPU [model] [options]
python train.py <method> <GPU> [model] [options]

# Using configuration file
python train.py <config.yaml>
```

#### Basic Examples

```bash
# PPO training (default CodeLlama-7b)
python train.py ppo 1

# GRPO training with DeepSeek model
python train.py grpo 0 deepseek-ai/deepseek-coder-6.7b-base

# PPO training with CodeLlama-13B
python train.py ppo 1 codellama/CodeLlama-13b-Python-hf

# Debug mode
python train.py ppo 1 --debug

# Disable WandB logging
python train.py grpo 0 --no-wandb
```

#### Advanced Options

```bash
# Custom batch size and dataset
python train.py ppo 1 --batch-size 8 --dataset humaneval

# Enable automatic evaluation after training
python train.py grpo 0 --auto-eval

# Custom number of epochs
python train.py ppo 1 --epochs 5

# Save merged model for each epoch
python train.py grpo 0 --save-merged-epochs

# Disable early stopping
python train.py ppo 1 --no-early-stopping

# Combined options
python train.py grpo 0 codellama/CodeLlama-13b-Python-hf \
    --dataset humaneval --batch-size 4 --epochs 3 --auto-eval
```

#### 🔬 Automatic Evaluation Feature

**Automatically evaluate MBPP and HumanEval datasets after training:**

```bash
# Enable automatic evaluation
python train.py ppo 1 --auto-eval

# Use DeepSeek model with automatic evaluation
python train.py grpo 0 deepseek-ai/deepseek-coder-6.7b-base --auto-eval

# Debug mode + automatic evaluation
python train.py ppo 1 --debug --auto-eval
```

**Automatic evaluation will:**
- ✅ **Evaluate both MBPP and HumanEval** datasets simultaneously
- ✅ Display key metrics like Pass@1, Pass@5 in command line
- ✅ Automatically log evaluation results to WandB
- ✅ Save detailed evaluation logs and result files
- ✅ Use the same standards as bigcode-evaluation-harness

**Evaluation configuration:**
- Temperature: 0.2 (reduce randomness for more stable results)
- Sample count: 15 (sufficient for accurate Pass@k metrics calculation)
- Batch size: 8 (balance speed and memory usage)
- Max length: 2048 tokens

#### Supported Models

```bash
# CodeLlama series
python train.py ppo 1 codellama/CodeLlama-7b-Python-hf    # Default
python train.py grpo 0 codellama/CodeLlama-13b-Python-hf

# DeepSeek series (automatically optimizes learning rate and batch size)
python train.py ppo 1 deepseek-ai/deepseek-coder-6.7b-base

# Qwen series (automatically optimizes generation length)
python train.py grpo 0 Qwen/Qwen2.5-Coder-7B

# Other models
python train.py ppo 1 microsoft/CodeT5p-770M
python train.py grpo 0 bigcode/starcoder2-15b
```

**Model-specific optimizations:**
- **DeepSeek**: Automatically reduces learning rate and batch size
- **CodeLlama-13B**: Adjusts batch size to accommodate larger model
- **Qwen**: Increases learning rate and generation length
- **Small models**: Increases batch size and learning rate

#### Supported Datasets

```bash
# MBPP (default)
python train.py ppo 1 --dataset mbpp

# HumanEval
python train.py grpo 0 --dataset humaneval
```

#### Naming Convention

New unified naming format: `{model_short_name}-{dataset}-{method}-qlora`

```
📁 Output directory: ./checkpoints/codellama-7b-python-mbpp-ppo-qlora/
🏷️ Hub model ID: codellama-7b-python-mbpp-ppo-qlora  
📊 WandB run: codellama-7b-python-mbpp-ppo-qlora-gpu1-0601-1234
🔗 Merged model: ./checkpoints/codellama-7b-python-mbpp-ppo-qlora-merged/
```

### Model Merging

Models are automatically merged after training. For manual merging:

```bash
# Manual adapter merging
python finetuning/merge_adapter.py \
    --adapter_model_id ./checkpoints/model-adapter \
    --output_dir ./merged_model \
    --device auto

# Upload to HuggingFace Hub
python finetuning/merge_adapter.py \
    --adapter_model_id ./checkpoints/model-adapter \
    --output_dir ./merged_model \
    --push_to_hub \
    --hub_model_id "your-username/model-name"
```

### Evaluation

Evaluation is integrated with bigcode-evaluation-harness:

```bash
# Manual evaluation (requires bigcode-evaluation-harness)
cd bigcode-evaluation-harness
accelerate launch main.py \
  --model /path/to/merged/model \
  --max_length_generation 512 \
  --tasks mbpp \
  --temperature 0.1 \
  --n_samples 15 \
  --batch_size 10 \
  --allow_code_execution
```

### Example Evaluation Results

After using `--auto-eval`, you will see similar output:

```
🎯 Automatic evaluation completed! Results summary:
============================================================

📊 MBPP Results:
   pass@1: 0.423
   pass@5: 0.578
   pass@10: 0.647

📊 HUMANEVAL Results:
   pass@1: 0.387
   pass@5: 0.542
   pass@10: 0.615

📁 Complete evaluation results saved to: evaluation_results_codellama-7b-python_ppo_20241201_143022.json
📊 Evaluation results logged to WandB: ['final_eval/mbpp_pass@1', 'final_eval/mbpp_pass@5', ...]
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
per_device_train_batch_size = 12
gradient_accumulation_steps = 1
learning_rate = 3e-6
num_train_epochs = 2
max_seq_length = 1024

# PPO specific
ppo_epochs = 2
cliprange = 0.15
target_kl = 0.3

# GRPO specific
num_samples = 8
kl_coeff = 0.02
clip_range = 0.2
```

### Reward Function

The reward function is designed to be result-oriented and dataset-specific:

```python
# Reward configuration
max_reward = 1.0          # For passing all test cases
min_reward = 0.0          # For failing test cases  
syntax_penalty = -0.2     # For syntax errors
quality_bonus = 0.1       # For code quality
timeout = 3.0             # Code execution timeout
```

## Dataset and Evaluation

### Supported Datasets

**MBPP (Mostly Basic Programming Problems)**
- **Training**: 374 problems from the train split
- **Evaluation**: 500 problems from the test split
- **Format**: InCoder-style prompts with docstrings and example tests

**HumanEval**
- **Training/Evaluation**: 164 problems from the test split
- **Format**: Function signature with docstring

### Evaluation Metrics

Evaluation follows the bigcode-evaluation-harness standard:
- **pass@k**: Percentage of problems solved with k attempts
- **Test Pass Rate**: Percentage of test cases that pass
- **Accuracy**: Percentage of problems with perfect solutions (single generation)
- **Success Rate**: Percentage of problems with positive rewards

## Advanced Usage

### Custom Configuration

Modify training parameters in `finetuning/config.py` or create custom configs:

```python
from finetuning.config import ExperimentConfig

# Create custom config
config = ExperimentConfig()
config.model.lora_r = 128
config.training.learning_rate = 1e-4
config.training.num_train_epochs = 5
```

### Multi-GPU Training

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES="1"
python train.py ppo 1

# The script automatically handles device mapping
```

### Memory Optimization

For limited GPU memory:

```bash
# Reduce batch size and increase accumulation
python train.py ppo 1 --batch-size 4 --debug

# Use smaller models
python train.py ppo 1 microsoft/CodeT5p-770M
```

## Project Structure

```
codegeneration/
├── finetuning/
│   ├── config.py              # Configuration classes
│   ├── data_utils.py           # Dataset processing and reward functions
│   ├── model_utils.py          # Model loading and QLoRA setup
│   ├── ppo_trainer.py          # PPO training implementation
│   ├── grpo_trainer.py         # GRPO training implementation
│   ├── train.py               # Main unified training script
│   ├── merge_adapter.py       # Adapter merging utilities
│   ├── evaluate.py            # Evaluation script
│   ├── requirements.txt       # Dependencies
│   └── README.md             # Detailed finetuning docs
├── bigcode-evaluation-harness/ # Evaluation harness (submodule)
└── README.md                  # This file
```

## Results and Benchmarks

Expected performance improvements over base models:

| Model | Method | MBPP pass@1 | HumanEval pass@1 |
|-------|--------|-------------|------------------|
| CodeLlama-7B (base) | - | ~30% | ~25% |
| CodeLlama-7B | PPO | ~35-40% | ~30-35% |
| CodeLlama-7B | GRPO | ~33-38% | ~28-33% |
| DeepSeek-Coder-6.7B | PPO | ~40-45% | ~35-40% |

*Note: Results may vary based on training configuration and hardware.*

## Monitoring and Logging

### Weights & Biases

Training automatically logs to W&B:

```bash
# Enable W&B logging (default)
python train.py ppo 1

# Disable W&B logging  
python train.py ppo 1 --no-wandb
```

Logged metrics include:
- Training loss (policy, value, entropy)
- Reward statistics
- Test pass rates
- KL divergence
- Early stopping metrics

### Local Logging

All outputs are logged to console and can be redirected:

```bash
# Save training logs
python train.py ppo 1 --auto-eval 2>&1 | tee training.log
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   python train.py ppo 1 --batch-size 4 --debug
   ```

2. **Model Loading Issues**:
   ```bash
   # Check HuggingFace token
   huggingface-cli login
   ```

3. **Code Execution Timeout**:
   ```python
   # Increase timeout in config.py
   config.training.code_execution_timeout = 10.0
   ```

4. **Early Stopping Too Aggressive**:
   ```bash
   # Disable early stopping
   python train.py ppo 1 --no-early-stopping
   ```

### Performance Tips

1. **Use appropriate batch sizes**: Start with small batches and increase gradually
2. **Monitor GPU utilization**: Use `nvidia-smi` to check resource usage
3. **Use auto-evaluation**: `--auto-eval` provides immediate feedback
4. **Save epoch models**: `--save-merged-epochs` for incremental evaluation

## Citation

If you use this code, please cite:

```bibtex
@misc{codellama-rl-finetuning,
  title={CodeLlama Reinforcement Learning Fine-tuning with QLoRA},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/codegeneration}
}
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [CodeLlama](https://github.com/facebookresearch/codellama) by Meta
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) by Google Research  
- [HumanEval](https://github.com/openai/human-eval) by OpenAI
- [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) by BigCode
- [TRL](https://github.com/huggingface/trl) by HuggingFace
- [PEFT](https://github.com/huggingface/peft) by HuggingFace 