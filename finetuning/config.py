import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """Model configuration for CodeLlama 7B with QLoRA."""
    model_name: str = "codellama/CodeLlama-7b-Python-hf"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # LoRA config
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Model settings
    max_seq_length: int = 1024
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    """Data configuration for MBPP dataset."""
    dataset_name: str = "mbpp"
    max_prompt_length: int = 512
    max_completion_length: int = 512
    train_test_split: float = 0.9
    num_proc: int = 4
    

@dataclass 
class TrainingConfig:
    """Training configuration for GRPO and PPO."""
    # Common training settings
    output_dir: str = "./checkpoints"
    per_device_train_batch_size: int = 12
    per_device_eval_batch_size: int = 12
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 2
    learning_rate: float = 3e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    
    # Evaluation and logging
    eval_steps: int = 250
    logging_steps: int = 5
    save_steps: int = 250
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_test_pass_rate"
    greater_is_better: bool = True
    
    # RL specific settings
    ppo_epochs: int = 2
    mini_batch_size: int = 1
    vf_coef: float = 0.3
    cliprange: float = 0.15
    cliprange_value: float = 0.15
    gamma: float = 1.0
    lam: float = 0.95
    target_kl: float = 0.5
    
    # GRPO specific settings
    grpo_alpha: float = 10.0
    
    # Generation settings
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Reward settings
    reward_model_path: Optional[str] = None
    code_execution_timeout: float = 10.0
    

@dataclass
class HuggingFaceConfig:
    """HuggingFace Hub configuration."""
    hub_model_id: str = "codellama-7b-mbpp-qlora"
    hub_token: Optional[str] = None
    push_to_hub: bool = True
    hub_private_repo: bool = False
    hub_strategy: str = "every_save"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hf: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    
    # Experiment settings
    seed: int = 42
    method: str = "ppo"  # "ppo" or "grpo"
    use_wandb: bool = True
    wandb_project: str = "codellama-mbpp-finetuning"
    wandb_run_name: Optional[str] = None
    
    def __post_init__(self):
        # Set wandb run name if not provided
        if self.wandb_run_name is None:
            self.wandb_run_name = f"codellama-7b-{self.method}-qlora"
        
        # Set output dir based on method
        self.training.output_dir = f"./checkpoints/{self.method}-qlora"
        
        # Set hub model id based on method
        self.hf.hub_model_id = f"codellama-7b-mbpp-{self.method}-qlora"


# Load configuration from environment variables
def load_config_from_env() -> ExperimentConfig:
    """Load configuration from environment variables."""
    config = ExperimentConfig()
    
    # HuggingFace token
    if "HF_TOKEN" in os.environ:
        config.hf.hub_token = os.environ["HF_TOKEN"]
    
    # Wandb settings
    if "WANDB_PROJECT" in os.environ:
        config.wandb_project = os.environ["WANDB_PROJECT"]
    
    # Method selection
    if "METHOD" in os.environ:
        config.method = os.environ["METHOD"].lower()
        assert config.method in ["ppo", "grpo"], f"Method must be 'ppo' or 'grpo', got {config.method}"
    
    return config 