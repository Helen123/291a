#!/usr/bin/env python3
"""
Unified reinforcement learning training script
Supports two usage modes:
1. Simple command line: python train.py ppo 1
2. YAML configuration: python train.py config.yaml

Features:
- PPO and GRPO training
- Automatic model merging
- Automatic evaluation command generation
"""

import os
import argparse
import logging
import torch
from pathlib import Path
import wandb
import sys
from datetime import datetime
import subprocess
import json
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import ExperimentConfig, load_config_from_env
from ppo_trainer import CodePPOTrainer
from grpo_trainer import CodeGRPOTrainer
from model_utils import calculate_model_size

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def show_usage():
    """Show usage instructions"""
    print("""
🚀 CodeLlama Fine-tuning Training Script

Usage:
Method 1: Simple command line mode
  python train.py <method> <gpu_id> [base_model] [options]

Parameter explanation:
  <method>      Fine-tuning method: ppo or grpo
  <gpu_id>      GPU number to use
  [base_model]  Optional, custom base model, e.g. codellama/CodeLlama-7b-Instruct-hf

Options:
  --debug       Debug mode, use small batches and frequent logging
  --no-wandb    Disable WandB logging
  --auto-eval   Enable automatic evaluation after training
  --batch-size N    Set batch size (e.g. --batch-size 4)
  --dataset NAME    Set dataset (mbpp or humaneval)
  --epochs N        Set number of epochs

Examples:
  # PPO training on GPU 0, using default model and MBPP dataset
  python train.py ppo 0

  # GRPO training on GPU 1, using custom model and HumanEval dataset
  python train.py grpo 1 deepseek-ai/deepseek-coder-6.7b-base --dataset humaneval

  # Use batch size 8, enable auto evaluation, train on MBPP
  python train.py ppo 1 --batch-size 8 --auto-eval --dataset mbpp

  # Debug mode, small batch training
  python train.py grpo 0 --debug --batch-size 2

  # Enable saving merged weights for each epoch, automatically clean original adapter
  python train.py ppo 1 --save-merged-epochs

  # Combined use of multiple parameters
  python train.py grpo 0 codellama/CodeLlama-13b-Python-hf --dataset humaneval --batch-size 4 --epochs 3

  # Disable early stopping mechanism, force completion of all epochs
  python train.py ppo 1 --no-early-stopping

  # Combined use: disable early stopping + save each epoch
  python train.py grpo 0 --no-early-stopping --save-merged-epochs

Method 2: YAML configuration file mode
  python train.py config.yaml

Configuration example:
  method: "ppo"
  gpu_id: "1"
  model:
    model_name: "codellama/CodeLlama-7b-Python-hf"
  training:
    per_device_train_batch_size: 4
    num_train_epochs: 3
    learning_rate: 1e-4
    """)


def extract_model_short_name(model_name: str) -> str:
    """
    从任意HuggingFace模型名称中提取简短名称
    
    Args:
        model_name: HuggingFace仓库名称，如 "microsoft/DialoGPT-medium"
        
    Returns:
        简短的模型名称，如 "dialogpt-medium"
    """
    if not model_name:
        return "unknown-model"
    
    # Extract the last part of repository name
    model_short_name = model_name.split('/')[-1]
    
    # Remove common suffixes
    common_suffixes = [
        "-hf", "-Python-hf", "-Instruct-hf", "-Chat-hf", 
        "-base", "-instruct", "-chat", "-code", "-text",
        "_base", "_instruct", "_chat", "_code", "_text"
    ]
    
    for suffix in common_suffixes:
        if model_short_name.endswith(suffix):
            model_short_name = model_short_name[:-len(suffix)]
            break
    
    # Convert to lowercase and replace special characters with hyphens
    model_short_name = model_short_name.lower()
    model_short_name = model_short_name.replace("_", "-")
    model_short_name = model_short_name.replace(" ", "-")
    
    # Limit length to avoid overly long directory names
    if len(model_short_name) > 25:
        model_short_name = model_short_name[:25].rstrip("-")
    
    return model_short_name


def setup_environment():
    """Setup environment variables and paths."""
    # Set HuggingFace cache directory
    os.environ.setdefault("HF_HOME", "./cache")
    
    # Set transformers cache
    os.environ.setdefault("TRANSFORMERS_CACHE", "./cache/transformers")
    
    # Allow code execution for reward function
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    
    # Create necessary directories
    Path("./cache").mkdir(exist_ok=True)
    Path("./checkpoints").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)


def create_config_from_args(method, gpu_id, base_model=None, debug_mode=False, no_wandb=False, auto_eval=False, batch_size=None, dataset=None, epochs=None, save_merged_epochs=False, no_early_stopping=False):
    """Create configuration from command line arguments"""
    config = ExperimentConfig()
    
    # Basic settings
    config.method = method
    config.use_wandb = not no_wandb
    config.wandb_project = "code-generation-rl"
    
    # Add auto_eval attribute to config
    config.auto_eval = auto_eval
    
    # Set dataset
    if dataset:
        if dataset.lower() not in ['mbpp', 'humaneval']:
            logger.error(f"❌ 不支持的数据集: {dataset}")
            logger.error("支持的数据集: mbpp, humaneval")
            sys.exit(1)
        config.data.dataset_name = dataset.lower()
        # HumanEval only has test split, MBPP defaults to using train split
        if dataset.lower() == 'humaneval':
            config.data.split = 'test'
        logger.info(f"📊 使用数据集: {dataset.upper()}")
    else:
        logger.info(f"📊 使用默认数据集: {config.data.dataset_name.upper()}")
    
    # Set base model
    if base_model:
        config.model.model_name = base_model
        logger.info(f"🤖 使用自定义模型: {base_model}")
    else:
        logger.info(f"🤖 使用默认模型: {config.model.model_name}")
    
    # Extract short name from model name for experiment identification
    actual_model_name = base_model if base_model else config.model.model_name
    model_short_name = extract_model_short_name(actual_model_name)
    
    # Debug mode configuration
    if debug_mode:
        config.training.logging_steps = 1
        config.training.save_steps = 5
        config.training.per_device_train_batch_size = 4
        logger.info("🐛 调试模式启用")
    
    # Auto evaluation mode
    if auto_eval:
        logger.info("🔬 自动评估模式启用: 训练完成后将自动评估MBPP和HumanEval")
    
    # Algorithm specific optimization configuration
    if method == 'ppo':
        config.training.learning_rate = 3e-6
        config.training.per_device_train_batch_size = 6  # Reduce batch size to accelerate training
        config.training.ppo_epochs = 2
        config.training.cliprange = 0.15
        config.training.target_kl = 0.5
        config.training.max_new_tokens = 1024  # Reduce generation length to accelerate training
    else:  # grpo
        config.training.learning_rate = 2e-6  # Lower GRPO learning rate to improve stability
        config.training.per_device_train_batch_size = 6  # Reduce batch size to accommodate more samples
        config.training.gradient_accumulation_steps = 2  # Compensate for reduced batch size
        # GRPO uses improved algorithm with enhanced clipping and KL regularization
    
    # Dataset specific optimization configuration
    if dataset and dataset.lower() == 'humaneval':
        # HumanEval dataset optimization - conservative but stable parameter settings
        if method == 'ppo':
            # PPO stable configuration for HumanEval
            config.training.learning_rate = 1.5e-6  # Significantly lower learning rate to prevent training collapse
            config.training.per_device_train_batch_size = 4  # Reduce back to 4 to improve stability
            config.training.gradient_accumulation_steps = 4  # Increase gradient accumulation to maintain effective batch size
            config.training.target_kl = 0.3  # Strict KL limit to prevent deviation too far
            config.training.cliprange = 0.15  # Lower clipping range, more conservative
            config.training.ppo_epochs = 2  # Lower PPO epochs to prevent overfitting
            config.training.temperature = 0.7  # Increase temperature to increase diversity
            config.training.max_new_tokens = 2048  # Maintain long generation length
            config.training.early_stopping_patience = 3  # Increase patience
        else:  # grpo
            # GRPO optimization for HumanEval
            config.training.learning_rate = 2e-6  # More conservative learning rate
            config.training.per_device_train_batch_size = 3  # Smaller batch size
            config.training.gradient_accumulation_steps = 4  # More gradient accumulation
            config.training.target_kl = 0.15  # Very strict KL limit
            config.training.cliprange = 0.1  # Smaller clipping range
            config.training.temperature = 0.7  # Moderate temperature
            config.training.max_new_tokens = 2048  # Long generation length
        
        # General HumanEval optimization
        config.training.warmup_steps = 10  # Increase warmup
        config.training.logging_steps = 2  # More frequent logging
        config.training.eval_steps = 5  # More frequent evaluation
        logger.info(f"🔧 HumanEval optimization: Using stable configuration to prevent training collapse, temperature={config.training.temperature}")
    
    elif dataset and dataset.lower() == 'mbpp':
        # MBPP dataset optimization - maintain default parameters
        pass
    
    # Model specific optimization configuration
    if base_model:
        model_name_lower = base_model.lower()
        
        # DeepSeek model optimization
        if "deepseek" in model_name_lower:
            config.training.learning_rate *= 0.7  # DeepSeek needs lower learning rate
            config.training.per_device_train_batch_size = max(4, config.training.per_device_train_batch_size // 2)
            config.training.gradient_accumulation_steps = 2
            logger.info("🔧 DeepSeek model optimization: Lower learning rate and batch size")
        
        # CodeLlama-13B optimization
        elif "13b" in model_name_lower and "codellama" in model_name_lower:
            config.training.per_device_train_batch_size = max(4, config.training.per_device_train_batch_size // 3)
            config.training.gradient_accumulation_steps = 3
            config.training.learning_rate *= 0.8
            logger.info("🔧 CodeLlama-13B optimization: Adjust batch size to accommodate larger model")
        
        # Qwen model optimization
        elif "qwen" in model_name_lower:
            config.training.learning_rate *= 1.2  # Qwen can use slightly higher learning rate
            config.training.max_new_tokens = 384  # Qwen has stronger generation capability
            logger.info("🔧 Qwen model optimization: Increase learning rate and generation length")
        
        # Small model optimization
        elif any(size in model_name_lower for size in ["770m", "1b", "3b"]):
            config.training.per_device_train_batch_size *= 2  # Small models can use larger batch size
            config.training.learning_rate *= 1.5  # Small models need higher learning rate
            logger.info("🔧 Small model optimization: Increase batch size and learning rate")
    
    # User specified batch_size has highest priority
    if batch_size is not None:
        config.training.per_device_train_batch_size = batch_size
        logger.info(f"✅ 使用用户指定的批大小: {batch_size}")
    
    # User specified epochs has highest priority
    if epochs is not None:
        config.training.num_train_epochs = epochs
        logger.info(f"✅ 使用用户指定的epoch数: {epochs}")
    
    # Save merged weights option
    config.save_merged_epochs = save_merged_epochs
    if save_merged_epochs:
        logger.info("💾 启用每个epoch保存合并权重功能")
    
    # Set output directory (new format: model-dataset-finetuning-method-qlora)
    dataset_name = config.data.dataset_name
    config.training.output_dir = f"./checkpoints/{model_short_name}-{dataset_name}-{method}-qlora"
    
    # Update other related naming
    config.hf.hub_model_id = f"{model_short_name}-{dataset_name}-{method}-qlora"
    
    # Set experiment name (consistent with output directory format)
    timestamp = datetime.now().strftime("%m%d-%H%M")
    config.wandb_run_name = f"{model_short_name}-{dataset_name}-{method}-qlora-gpu{gpu_id}-{timestamp}"
    
    # Add no_early_stopping to config
    config.no_early_stopping = no_early_stopping
    if no_early_stopping:
        logger.info("🔄 启用禁用早停机制，强制完成所有epoch")
    
    return config


def parse_arguments():
    """Parse command line arguments"""
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(0)
    
    first_arg = sys.argv[1]
    
    # Check if it's help
    if first_arg in ['-h', '--help']:
        show_usage()
        sys.exit(0)
    
    # Check if it's config file (contains .yaml or .yml)
    if '.yaml' in first_arg or '.yml' in first_arg:
        return 'config_file', first_arg, {}
    
    # Otherwise parse as simple command line mode
    if len(sys.argv) < 3:
        logger.error("❌ 简单模式需要指定方法和GPU")
        show_usage()
        sys.exit(1)
    
    method = sys.argv[1].lower()
    gpu_id = sys.argv[2]
    
    # Validate parameters
    if method not in ['ppo', 'grpo']:
        logger.error(f"❌ 不支持的方法: {method}")
        logger.error("支持的方法: ppo, grpo")
        sys.exit(1)
    
    try:
        int(gpu_id)  # Validate GPU ID is a number
    except ValueError:
        logger.error(f"❌ 无效的GPU ID: {gpu_id}")
        sys.exit(1)
    
    # Check if third parameter is base model
    base_model = None
    option_start_idx = 3
    
    if len(sys.argv) > 3 and not sys.argv[3].startswith('--'):
        # Third parameter is not an option, consider it as base model
        base_model = sys.argv[3]
        option_start_idx = 4
        logger.info(f"🤖 检测到自定义模型: {base_model}")
    
    # Parse options (start from correct position)
    options = {
        'base_model': base_model,
        'debug_mode': '--debug' in sys.argv[option_start_idx:],
        'no_wandb': '--no-wandb' in sys.argv[option_start_idx:],
        'auto_eval': '--auto-eval' in sys.argv[option_start_idx:],
        'batch_size': None,
        'dataset': None,
        'epochs': None,  # Add epochs option
        'save_merged_epochs': '--save-merged-epochs' in sys.argv[option_start_idx:],  # Add save merged weights option
        'no_early_stopping': '--no-early-stopping' in sys.argv[option_start_idx:]  # Add no_early_stopping option
    }
    
    # Parse parameter values
    for i, arg in enumerate(sys.argv[option_start_idx:], start=option_start_idx):
        # Parse --batch-size parameter
        if arg == '--batch-size' and i + 1 < len(sys.argv):
            try:
                options['batch_size'] = int(sys.argv[i + 1])
                logger.info(f"🎯 检测到自定义批大小: {options['batch_size']}")
            except ValueError:
                logger.error(f"❌ 无效的批大小值: {sys.argv[i + 1]}")
                sys.exit(1)
        
        # Parse --dataset parameter
        elif arg == '--dataset' and i + 1 < len(sys.argv):
            options['dataset'] = sys.argv[i + 1]
            logger.info(f"📊 检测到数据集参数: {options['dataset']}")
        
        # Parse --epochs parameter
        elif arg == '--epochs' and i + 1 < len(sys.argv):
            try:
                options['epochs'] = int(sys.argv[i + 1])
                logger.info(f"🔄 检测到自定义epoch数: {options['epochs']}")
            except ValueError:
                logger.error(f"❌ 无效的epoch数值: {sys.argv[i + 1]}")
                sys.exit(1)
    
    return 'simple', (method, gpu_id), options


def merge_adapter_to_base(adapter_path: str, base_model_name: str, merged_output_dir: str) -> str:
    """
    Merge LoRA adapter to base model.
    
    Args:
        adapter_path: Path to the adapter model
        base_model_name: Name of the base model
        merged_output_dir: Directory to save merged model
        
    Returns:
        Path to the merged model
    """
    logger.info(f"🔧 Loading base model: {base_model_name}")
    
    # First load adapter's tokenizer to get correct vocabulary size
    logger.info(f"🔍 检查adapter tokenizer: {adapter_path}")
    try:
        adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        adapter_vocab_size = len(adapter_tokenizer)
        logger.info(f"📊 Adapter词汇表大小: {adapter_vocab_size}")
    except Exception as e:
        logger.warning(f"⚠️  无法从adapter路径加载tokenizer ({e})，将使用基础模型tokenizer")
        adapter_tokenizer = None
        adapter_vocab_size = None
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_vocab_size = len(base_tokenizer)
    logger.info(f"📊 基础模型词汇表大小: {base_vocab_size}")
    
    # If vocabulary sizes don't match, need to adjust base model
    if adapter_vocab_size is not None and adapter_vocab_size != base_vocab_size:
        logger.info(f"⚠️  词汇表大小不匹配! 调整基础模型词汇表: {base_vocab_size} -> {adapter_vocab_size}")
        base_model.resize_token_embeddings(adapter_vocab_size)
        logger.info("✅ 基础模型词汇表已调整")
    
    logger.info(f"🔗 Loading adapter from: {adapter_path}")
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("🔄 Merging adapter to base model...")
    
    # Merge adapter weights to base model
    merged_model = model.merge_and_unload()
    
    logger.info(f"💾 Saving merged model to: {merged_output_dir}")
    
    # Create output directory
    Path(merged_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save merged model
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    
    # Use adapter's tokenizer (if available), otherwise use base tokenizer
    if adapter_tokenizer is not None:
        adapter_tokenizer.save_pretrained(merged_output_dir)
        logger.info("✅ 使用adapter tokenizer保存")
    else:
        base_tokenizer.save_pretrained(merged_output_dir)
        logger.info("✅ 使用基础模型tokenizer保存")
    
    logger.info("✅ Model merging completed successfully!")
    
    return merged_output_dir


def validate_evaluation_environment():
    """
    Validate evaluation environment is available
    
    Returns:
        bool: Whether evaluation can be performed
    """
    eval_harness_path = Path("../bigcode-evaluation-harness")
    
    # Check directory exists
    if not eval_harness_path.exists():
        logger.error("❌ bigcode-evaluation-harness目录不存在")
        return False
    
    # Check main.py exists
    main_py = eval_harness_path / "main.py"
    if not main_py.exists():
        logger.error("❌ bigcode-evaluation-harness/main.py不存在")
        return False
    
    # Check if accelerate command is available
    try:
        result = subprocess.run(["accelerate", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.error("❌ accelerate命令不可用")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("❌ accelerate命令不可用或超时")
        return False
    
    logger.info("✅ 评估环境验证通过")
    return True


def auto_evaluate_model(merged_model_path: str, model_short_name: str, method: str) -> Dict[str, Any]:
    """
    Auto evaluate merged model's performance on MBPP and HumanEval
    
    Args:
        merged_model_path: Path to merged model
        model_short_name: Model short name
        method: Training method (ppo/grpo)
        
    Returns:
        Evaluation results dictionary
    """
    logger.info("🔬 开始自动评估模型性能...")
    
    # Validate evaluation environment
    if not validate_evaluation_environment():
        logger.error("❌ 评估环境验证失败，跳过自动评估")
        return {}
    
    eval_harness_path = Path("../bigcode-evaluation-harness")
    
    # Evaluation configuration
    eval_configs = [
        {
            "task": "mbpp",
            "max_length": 2048,
            "temperature": 0.2,
            "n_samples": 15,  # Use more samples for more accurate Pass@k
            "batch_size": 8
        },
        {
            "task": "humaneval", 
            "max_length": 2048,
            "temperature": 0.2,
            "n_samples": 15,
            "batch_size": 8
        }
    ]
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config in eval_configs:
        task = config["task"]
        logger.info(f"📊 评估 {task.upper()} 数据集...")
        
        # Build evaluation command
        cmd = [
            "accelerate", "launch", "main.py",
            "--model", str(merged_model_path),
            "--max_length_generation", str(config["max_length"]),
            "--tasks", task,
            "--temperature", str(config["temperature"]),
            "--n_samples", str(config["n_samples"]),
            "--batch_size", str(config["batch_size"]),
            "--allow_code_execution"
        ]
        
        # Set output file
        output_file = f"evaluation_{model_short_name}_{method}_{task}_{timestamp}.json"
        
        try:
            logger.info(f"🚀 运行评估命令: {' '.join(cmd)}")
            
            # Switch to evaluation directory and run command
            original_dir = os.getcwd()
            os.chdir(eval_harness_path)
            
            # Run evaluation
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Switch back to original directory
            os.chdir(original_dir)
            
            if process.returncode == 0:
                logger.info(f"✅ {task.upper()} 评估完成")
                
                # Try to parse results from output
                output_lines = process.stdout.split('\n')
                task_metrics = {}
                for line in output_lines:
                    line = line.strip()
                    if "pass@" in line.lower():
                        logger.info(f"📈 {task.upper()} 结果: {line}")
                        # Try to extract numerical values
                        if ":" in line:
                            parts = line.split(":")
                            if len(parts) >= 2:
                                metric_name = parts[0].strip()
                                try:
                                    metric_value = float(parts[1].strip().rstrip('%')) / 100.0
                                    task_metrics[metric_name] = metric_value
                                except ValueError:
                                    pass
                
                # If results were parsed, save to results
                if task_metrics:
                    results[task] = task_metrics
                
                # Save output to log file
                log_file = f"logs/eval_{model_short_name}_{method}_{task}_{timestamp}.log"
                Path("logs").mkdir(exist_ok=True)
                with open(log_file, 'w') as f:
                    f.write(f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}\n")
                    
            else:
                logger.error(f"❌ {task.upper()} 评估失败")
                logger.error(f"错误输出: {process.stderr}")
                
                # Save error log
                error_log = f"logs/eval_error_{model_short_name}_{method}_{task}_{timestamp}.log"
                Path("logs").mkdir(exist_ok=True)
                with open(error_log, 'w') as f:
                    f.write(f"Return code: {process.returncode}\n")
                    f.write(f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}\n")
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {task.upper()} 评估超时")
            os.chdir(original_dir)  # Ensure switch back to original directory
        except Exception as e:
            logger.error(f"💥 {task.upper()} 评估出现异常: {e}")
            os.chdir(original_dir)  # Ensure switch back to original directory
    
    # Generate evaluation summary
    if results:
        logger.info("=" * 80)
        logger.info("🎯 自动评估完成! 结果总结:")
        logger.info("=" * 80)
        
        for task, task_results in results.items():
            logger.info(f"\n📊 {task.upper()} 结果:")
            # Try to extract key metrics
            if isinstance(task_results, dict):
                for key, value in task_results.items():
                    if "pass@" in key.lower():
                        logger.info(f"   {key}: {value}")
            
        # Save full results
        final_results_file = f"evaluation_results_{model_short_name}_{method}_{timestamp}.json"
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📁 完整评估结果已保存至: {final_results_file}")
        
    else:
        logger.warning("⚠️ 未获得有效的评估结果")
        
    return results


def run_training_experiment(config: ExperimentConfig):
    """
    Run complete training experiment with automatic adapter merging.
    
    Args:
        config: Experiment configuration
    """
    logger.info("=" * 80)
    logger.info("🚀 STARTING CODELLAMA FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"📊 算法: {config.method.upper()}")
    logger.info(f"🤖 模型: {config.model.model_name}")
    logger.info(f"📂 输出: {config.training.output_dir}")
    logger.info(f"🔄 学习率: {config.training.learning_rate}")
    logger.info(f"📦 批大小: {config.training.per_device_train_batch_size}")
    logger.info(f"📈 轮数: {config.training.num_train_epochs}")
    
    # Show algorithm specific parameters
    if config.method == 'ppo':
        logger.info(f"⚙️ PPO轮数: {config.training.ppo_epochs}")
        logger.info(f"🎯 目标KL: {config.training.target_kl}")
    else:
        logger.info("⚙️ GRPO uses improved algorithm with enhanced clipping and KL regularization")
    
    # Initialize WandB if enabled
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__
        )
        logger.info(f"📊 WandB logging initialized: {config.wandb_project}")
    
    try:
        # Initialize trainer based on method
        if config.method.upper() == "PPO":
            logger.info("🏃 Initializing PPO trainer...")
            trainer = CodePPOTrainer(config)
        elif config.method.upper() == "GRPO":
            logger.info("🏃 Initializing GRPO trainer...")
            trainer = CodeGRPOTrainer(config)
        else:
            raise ValueError(f"Unknown training method: {config.method}")
        
        # Start training
        logger.info("🎯 Starting training...")
        training_results = trainer.train()
        
        logger.info("=" * 80)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        # Log final training metrics
        if training_results and isinstance(training_results, dict):
            for key, value in training_results.items():
                logger.info(f"{key}: {value}")
        
        # Get adapter path (should be saved in output directory)
        adapter_path = Path(config.training.output_dir)
        
        # Check if adapter was saved
        if not (adapter_path / "final_model").exists():
            logger.warning("⚠️ No final_model found, checking for alternative save formats...")
            if not any(adapter_path.glob("**/adapter_model.safetensors")):
                logger.error("❌ No model files found in output directory")
                return
            # Use best_epoch_model if final_model doesn't exist
            if (adapter_path / "best_epoch_model").exists():
                adapter_path = adapter_path / "best_epoch_model"
            else:
                adapter_path = adapter_path / "best_model"
        else:
            adapter_path = adapter_path / "final_model"
        
        # AUTO-MERGE ADAPTER TO BASE MODEL
        logger.info("🔗 AUTO-MERGING ADAPTER TO BASE MODEL...")
        
        # Create merged model directory with new naming format
        # Extract model information from config
        base_model_name = config.model.model_name
        method = config.method
        dataset_name = config.data.dataset_name
        
        # Extract model short name
        model_short_name = extract_model_short_name(base_model_name)
        
        # New merged model naming format: model-dataset-finetuning-method-qlora-merged
        merged_model_dir = f"./checkpoints/{model_short_name}-{dataset_name}-{method}-qlora-merged"
        
        try:
            merged_path = merge_adapter_to_base(
                adapter_path=str(adapter_path),
                base_model_name=config.model.model_name,
                merged_output_dir=merged_model_dir
            )
            
            logger.info("=" * 80)
            logger.info("🎯 EXPERIMENT COMPLETED WITH AUTO-MERGE!")
            logger.info("=" * 80)
            logger.info(f"📁 Adapter saved at: {adapter_path}")
            logger.info(f"📁 Merged model saved at: {merged_path}")
            
            # Log evaluation command
            logger.info("\n📋 EVALUATION COMMANDS:")
            logger.info("-" * 50)
            
            # Evaluation with our compatible script
            logger.info("1. 使用本地评估脚本:")
            logger.info(f"python evaluate.py {merged_path} --num_samples 15 --temperature 0.1")
            
            # Evaluation with official bigcode-evaluation-harness
            logger.info("\n2. 使用官方bigcode-evaluation-harness:")
            logger.info(f"""cd ../bigcode-evaluation-harness
accelerate launch main.py \\
  --model {os.path.abspath(merged_path)} \\
  --max_length_generation 512 \\
  --tasks mbpp \\
  --temperature 0.1 \\
  --n_samples 15 \\
  --batch_size 10 \\
  --allow_code_execution""")
            
            # Auto-evaluate merged model if enabled
            if getattr(config, 'auto_eval', False):
                logger.info("🔬 开始自动评估合并后的模型...")
                auto_eval_results = auto_evaluate_model(merged_path, model_short_name, method)
                
                # Log evaluation results to WandB
                if config.use_wandb and auto_eval_results:
                    wandb_eval_results = {}
                    for task, task_results in auto_eval_results.items():
                        if isinstance(task_results, dict):
                            for key, value in task_results.items():
                                if "pass@" in key.lower():
                                    wandb_key = f"final_eval/{task}_{key}"
                                    wandb_eval_results[wandb_key] = value
                
                    if wandb_eval_results:
                        wandb.log(wandb_eval_results)
                        logger.info(f"📊 评估结果已记录到WandB: {list(wandb_eval_results.keys())}")
            else:
                logger.info("ℹ️  自动评估未启用。使用 --auto-eval 参数启用自动评估功能。")
            
            logger.info("=" * 80)
            logger.info("🏆 COMPLETE TRAINING AND EVALUATION FINISHED!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Failed to merge adapter: {e}")
            logger.info("🔧 You can manually merge later using:")
            logger.info(f"python merge_adapter.py --adapter_model_id {adapter_path} --output_dir {merged_model_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error("Please check the logs above for details")
        if config.use_wandb:
            wandb.finish(exit_code=1)
        raise
    
    finally:
        # Clean up WandB
        if config.use_wandb:
            wandb.finish()


def main():
    """Main function"""
    setup_environment()
    
    # Parse arguments
    mode, args, options = parse_arguments()
    
    try:
        if mode == 'config_file':
            # YAML configuration file mode
            config_file = args
            logger.info(f"✅ 使用配置文件模式: {config_file}")
            
            if not Path(config_file).exists():
                logger.error(f"❌ 配置文件不存在: {config_file}")
                sys.exit(1)
            
            # Here we need to implement YAML loading, temporarily using environment variable mode
            config = load_config_from_env()
            logger.info(f"✅ Configuration loaded from environment")
            
        else:
            # Simple command line mode
            method, gpu_id = args
            logger.info(f"⚡ 使用简单命令行模式: {method.upper()} on GPU {gpu_id}")
            
            # Set GPU environment
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            logger.info(f"🎯 设置GPU: {gpu_id}")
            
            # Create configuration
            config = create_config_from_args(method, gpu_id, **options)
        
        # Run experiment
        run_training_experiment(config)
        logger.info("🎉 完整实验流程成功完成!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ 训练被用户中断")
    except Exception as e:
        logger.error(f"💥 实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 