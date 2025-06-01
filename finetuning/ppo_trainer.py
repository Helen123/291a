import torch
import logging
import wandb
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from transformers import TrainingArguments
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim

from config import ExperimentConfig
from model_utils import load_tokenizer, load_base_model, setup_lora_model, save_model_and_tokenizer, calculate_model_size
from data_utils import create_mbpp_dataset_for_rl, RewardFunction, MBPPDataProcessor

logger = logging.getLogger(__name__)


class CodePPOTrainer:
    """Complete PPO Trainer for code generation with gradient updates."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize PPO trainer.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.reward_fn = None
        self.dataset = None
        self.optimizer = None
        
        # IMPROVED PPO hyperparameters (根据训练结果优化)
        self.ppo_epochs = 2  # 减少PPO epochs，防止过拟合
        self.clip_range = 0.15  # 稍微降低clip range，更保守的更新
        self.value_loss_coef = 0.3  # 降低value loss权重
        self.entropy_loss_coef = 0.02  # 增加熵损失，鼓励探索
        self.kl_coef = 0.15  # 大幅增加KL惩罚，防止偏离太远
        self.learning_rate = 3e-6  # 大幅降低学习率，更稳定
        self.max_kl_threshold = 0.5  # 添加KL阈值，超过则跳过更新
        
        # 早停和最佳模型跟踪
        self.best_test_pass_rate = 0.0
        self.best_reward = 0.0
        self.patience = 2  # 连续2个epoch性能下降则停止
        self.patience_counter = 0
        self.early_stop = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def setup_model(self):
        """Setup model and tokenizer for PPO training."""
        logger.info("Loading model and tokenizer for PPO training...")
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(self.config.model)
        
        # Fix attention mask issue by setting a proper pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        
        # Ensure pad_token is different from eos_token
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            # Use a different token for padding
            self.tokenizer.pad_token = "<pad>"
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        # Load base model and setup LoRA
        base_model = load_base_model(self.config.model)
        base_model = setup_lora_model(base_model, self.config.model)
        
        # Resize token embeddings if we added new tokens
        if len(self.tokenizer) > base_model.config.vocab_size:
            base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Wrap with value head for policy model
        self.model = AutoModelForCausalLMWithValueHead(base_model)
        
        # Fix TRL compatibility issue - manually set is_peft_model attribute
        if hasattr(base_model, 'peft_config'):
            self.model.is_peft_model = True
        else:
            self.model.is_peft_model = False
        
        # Create reference model (frozen copy) - just the base model without value head
        ref_base_model = load_base_model(self.config.model)
        ref_base_model = setup_lora_model(ref_base_model, self.config.model)
        
        # Resize reference model embeddings too
        if len(self.tokenizer) > ref_base_model.config.vocab_size:
            ref_base_model.resize_token_embeddings(len(self.tokenizer))
            
        self.ref_model = ref_base_model.eval()
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Setup optimizer with improved settings
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params, 
            lr=self.learning_rate,
            betas=(0.9, 0.95),  # 更保守的momentum
            eps=1e-8,
            weight_decay=0.01  # 添加权重衰减
        )
        
        # Calculate and log model size
        model_stats = calculate_model_size(self.model)
        logger.info(f"Policy model loaded with {model_stats['trainable_params']:,} trainable parameters")
        logger.info("Reference model created (frozen copy)")
        logger.info(f"Optimizer initialized with learning rate: {self.learning_rate}")
        logger.info(f"IMPROVED PPO hyperparameters:")
        logger.info(f"  PPO epochs: {self.ppo_epochs}")
        logger.info(f"  Clip range: {self.clip_range}")
        logger.info(f"  KL coefficient: {self.kl_coef}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Max KL threshold: {self.max_kl_threshold}")
        logger.info(f"  Early stopping patience: {self.patience}")
        logger.info(f"PEFT model detected: {self.model.is_peft_model}")
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        logger.info(f"EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        
    def setup_data(self):
        """Setup training dataset."""
        logger.info("Loading and preprocessing MBPP dataset...")
        
        # Load dataset
        self.dataset = create_mbpp_dataset_for_rl(
            split="train",
            max_samples=None  # Use full dataset
        )
        
        # Filter out examples that are too long
        def filter_long_examples(example):
            prompt_length = len(self.tokenizer.encode(example["prompt"]))
            return prompt_length <= self.config.data.max_prompt_length
        
        self.dataset = self.dataset.filter(filter_long_examples)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return {
                "input_ids": [
                    self.tokenizer.encode(prompt, truncation=True, max_length=self.config.data.max_prompt_length)
                    for prompt in examples["prompt"]
                ]
            }
        
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        
        logger.info(f"Dataset loaded with {len(self.dataset)} examples")
        
    def setup_reward_function(self):
        """Setup reward function for code execution."""
        logger.info("Setting up reward function...")
        
        self.reward_fn = RewardFunction(
            timeout=self.config.training.code_execution_timeout,
            max_reward=1.0,
            min_reward=0.0,
            syntax_penalty=-0.1
        )
        
    def generate_response(self, query_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate response for a given query.
        
        Args:
            query_tensor: Input query tensor
            
        Returns:
            Generated response tensor
        """
        # Ensure query_tensor is on the same device as model
        device = next(self.model.parameters()).device
        query_tensor = query_tensor.to(device)
        
        # Create attention mask
        attention_mask = torch.ones_like(query_tensor)
        
        generation_kwargs = {
            "max_new_tokens": self.config.training.max_new_tokens,
            "do_sample": self.config.training.do_sample,
            "temperature": self.config.training.temperature,
            "top_p": self.config.training.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }
        
        # Generate using the base model
        with torch.no_grad():
            response_tensor = self.model.generate(
                query_tensor,
                **generation_kwargs
            )
        
        # Extract only the generated part (remove the prompt)
        response_only = response_tensor[:, query_tensor.shape[1]:]
        
        return response_only
    
    def compute_kl_divergence(self, query_tensors: List[torch.Tensor], 
                            response_tensors: List[torch.Tensor]) -> List[float]:
        """
        Compute KL divergence between policy and reference model.
        
        Args:
            query_tensors: List of query tensors
            response_tensors: List of response tensors
            
        Returns:
            List of KL divergences
        """
        kl_divergences = []
        
        for query_tensor, response_tensor in zip(query_tensors, response_tensors):
            # Combine query and response
            full_sequence = torch.cat([query_tensor, response_tensor], dim=-1)
            
            # Get logits from policy model
            with torch.no_grad():
                policy_outputs = self.model.pretrained_model(full_sequence)
                policy_logits = policy_outputs.logits
                
                # Get logits from reference model
                ref_outputs = self.ref_model(full_sequence)
                ref_logits = ref_outputs.logits
                
                # Compute log probabilities
                policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                
                # Compute KL divergence for response tokens only
                response_start = query_tensor.shape[-1]
                response_policy_log_probs = policy_log_probs[:, response_start-1:-1, :]
                response_ref_log_probs = ref_log_probs[:, response_start-1:-1, :]
                
                # KL(policy || ref) = sum(policy_prob * (log(policy_prob) - log(ref_prob)))
                kl = F.kl_div(response_ref_log_probs, response_policy_log_probs, 
                             reduction='none', log_target=True)
                kl_mean = kl.sum(dim=-1).mean().item()
                kl_divergences.append(kl_mean)
        
        return kl_divergences

    def compute_rewards(self, queries: List[str], responses: List[str], 
                       test_cases: List[str], query_tensors: List[torch.Tensor],
                       response_tensors: List[torch.Tensor]) -> Tuple[List[float], Dict[str, float]]:
        """
        Compute rewards for generated responses with KL penalty.
        Uses EXACT same evaluation as bigcode-evaluation-harness during training.
        
        Args:
            queries: List of input queries (prompts)
            responses: List of generated responses (raw model outputs)
            test_cases: List of test cases for evaluation
            query_tensors: List of query tensors
            response_tensors: List of response tensors
            
        Returns:
            Tuple of (reward_values, detailed_metrics)
        """
        # Combine queries and responses to get full generated sequences
        full_generations = []
        for query, response in zip(queries, responses):
            # Decode the full sequence if needed
            if isinstance(response, torch.Tensor):
                response_text = self.tokenizer.decode(response, skip_special_tokens=True)
            else:
                response_text = response
            
            # The full generation includes prompt + generated part
            full_generation = query + response_text
            full_generations.append(full_generation)
        
        # Compute detailed rewards using BIGCODE-COMPATIBLE processing
        base_rewards, details = self.reward_fn.compute_batch_rewards_detailed(
            generated_codes=full_generations,
            prompts=queries,
            test_cases=test_cases,
            max_workers=4
        )
        
        # Compute KL divergences (only if we have tensor data)
        kl_divergences = []
        if query_tensors and response_tensors:
            kl_divergences = self.compute_kl_divergence(query_tensors, response_tensors)
        else:
            kl_divergences = [0.0] * len(base_rewards)
        
        # Apply KL penalty (standard PPO approach)
        final_rewards = []
        for base_reward, kl_div in zip(base_rewards, kl_divergences):
            penalized_reward = base_reward - self.kl_coef * kl_div
            final_rewards.append(penalized_reward)
        
        # Compute aggregate metrics for logging
        valid_details = [d for d in details if d and not d.get('has_syntax_error', False)]
        
        if valid_details:
            avg_test_ratio = sum(d.get('test_ratio', 0.0) for d in valid_details) / len(valid_details)
            avg_quality_score = sum(d.get('quality_score', 0.0) for d in valid_details) / len(valid_details)
            total_passed = sum(d.get('passed_tests', 0) for d in details)
            total_tests = sum(d.get('total_tests', 0) for d in details)
        else:
            avg_test_ratio = 0.0
            avg_quality_score = 0.0
            total_passed = 0
            total_tests = 0
        
        syntax_errors = sum(1 for d in details if d.get('has_syntax_error', False))
        
        # Count problems with perfect solutions (pass_ratio = 1.0) for bigcode-style reporting
        perfect_solutions = sum(1 for d in details if d.get('test_ratio', 0.0) == 1.0)
        
        metrics = {
            'avg_test_ratio': avg_test_ratio,
            'avg_quality_score': avg_quality_score,
            'total_passed_tests': total_passed,
            'total_tests': total_tests,
            'syntax_errors': syntax_errors,
            'test_pass_rate': total_passed / total_tests if total_tests > 0 else 0.0,
            'perfect_solutions': perfect_solutions,
            'perfect_solution_rate': perfect_solutions / len(details) if details else 0.0,
            'avg_kl_divergence': sum(kl_divergences) / len(kl_divergences) if kl_divergences else 0.0,
            'avg_base_reward': sum(base_rewards) / len(base_rewards) if base_rewards else 0.0,
            'avg_final_reward': sum(final_rewards) / len(final_rewards) if final_rewards else 0.0,
        }
        
        return final_rewards, metrics
        
    def compute_policy_loss(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor, 
                           advantages: torch.Tensor, clip_range: float = 0.2) -> torch.Tensor:
        """
        Compute PPO clipped policy loss.
        
        Args:
            old_log_probs: Log probabilities from old policy (detached)
            new_log_probs: Log probabilities from current policy
            advantages: Advantage estimates
            clip_range: PPO clipping parameter
            
        Returns:
            Policy loss tensor
        """
        # Compute probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Compute clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        
        return policy_loss
    
    def compute_value_loss(self, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute value function loss.
        
        Args:
            values: Predicted values from value head
            returns: Target returns
            
        Returns:
            Value loss tensor
        """
        return F.mse_loss(values, returns)
    
    def compute_log_probs(self, model: torch.nn.Module, input_ids: torch.Tensor, 
                         attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute log probabilities for given input sequences.
        
        Args:
            model: Model to use for computation
            input_ids: Input token ids
            attention_mask: Attention mask
            
        Returns:
            Log probabilities for each token
        """
        if hasattr(model, 'pretrained_model'):
            # Use the base model from AutoModelForCausalLMWithValueHead
            outputs = model.pretrained_model(input_ids, attention_mask=attention_mask)
        else:
            # Direct model call
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log probabilities for actual tokens (shift by 1)
        log_probs_tokens = log_probs[:, :-1, :].gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        return log_probs_tokens
    
    def compute_advantages_and_returns(self, rewards: List[float], values: torch.Tensor,
                                     gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation) and returns.
        
        Args:
            rewards: List of rewards for each step
            values: Value predictions from value head
            gamma: Discount factor
            lam: GAE lambda parameter
            
        Returns:
            Tuple of (advantages, returns)
        """
        device = values.device
        rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        
        # For single reward case, create simple advantage and return
        if len(rewards) == 1:
            # Simple case: single reward
            advantage = rewards_tensor[0]  # Use reward as advantage
            return_val = rewards_tensor[0]  # Use reward as return
            return advantage.unsqueeze(0), return_val.unsqueeze(0)
        
        # For multiple rewards, use proper GAE
        # Compute returns (discounted rewards)
        returns = torch.zeros_like(rewards_tensor)
        returns[-1] = rewards_tensor[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards_tensor[t] + gamma * returns[t + 1]
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards_tensor)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextvalue = 0
            else:
                nextvalue = values[t + 1] if values.numel() > t + 1 else 0
            
            current_value = values[t] if values.numel() > t else 0
            delta = rewards_tensor[t] + gamma * nextvalue - current_value
            advantages[t] = lastgaelam = delta + gamma * lam * lastgaelam
        
        return advantages, returns

    def train_step_complete_ppo(self, batch_data):
        """Complete PPO training step with gradient updates and KL divergence check."""
        # Extract batch data
        queries = batch_data["prompt"]
        test_cases = batch_data["test_cases"]
        
        # Get model device
        device = next(self.model.parameters()).device
        
        # Tokenize queries with proper formatting
        query_tensors = []
        for query in queries:
            # Tokenize with attention mask
            encoded = self.tokenizer(
                query, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=False
            )
            query_tensor = encoded['input_ids'].to(device)
            query_tensors.append(query_tensor)
        
        # Generate responses and collect old log probabilities
        responses = []
        response_tensors = []
        old_log_probs_list = []
        old_values_list = []
        
        self.model.eval()  # Set to eval mode for generation
        with torch.no_grad():
            for query_tensor in query_tensors:
                try:
                    # Generate response
                    response_tensor = self.generate_response(query_tensor)
                    response = self.tokenizer.decode(response_tensor[0], skip_special_tokens=True)
                    responses.append(response)
                    response_tensors.append(response_tensor)
                    
                    # Compute old log probabilities and values
                    full_sequence = torch.cat([query_tensor, response_tensor], dim=-1)
                    attention_mask = torch.ones_like(full_sequence)
                    
                    # Get log probabilities from current model (will be "old" after we update)
                    old_log_probs = self.compute_log_probs(self.model, full_sequence, attention_mask)
                    old_log_probs_list.append(old_log_probs)
                    
                    # Get value predictions
                    model_outputs = self.model(full_sequence, attention_mask=attention_mask)
                    if hasattr(model_outputs, 'value') and model_outputs.value is not None:
                        values = model_outputs.value.squeeze(-1)  # Remove last dim
                    elif isinstance(model_outputs, tuple) and len(model_outputs) >= 2:
                        # Handle tuple output format (logits, values, ...)
                        logits, values = model_outputs[0], model_outputs[1]
                        if values is not None:
                            values = values.squeeze(-1)
                        else:
                            values = torch.zeros((full_sequence.shape[0], full_sequence.shape[1]), device=device)
                    else:
                        # Fallback: create zero values
                        values = torch.zeros((full_sequence.shape[0], full_sequence.shape[1]), device=device)
                    old_values_list.append(values)
                    
                except Exception as e:
                    logger.warning(f"Generation failed for query: {e}")
                    responses.append("")
                    # Create empty tensors for failed generation
                    empty_response = torch.zeros((1, 1), device=device, dtype=torch.long)
                    empty_log_probs = torch.zeros((1, 1), device=device, dtype=torch.float32)
                    empty_values = torch.zeros((1, 1), device=device, dtype=torch.float32)
                    response_tensors.append(empty_response)
                    old_log_probs_list.append(empty_log_probs)
                    old_values_list.append(empty_values)
        
        # Compute rewards with KL penalty
        rewards, metrics = self.compute_rewards(queries, responses, test_cases, 
                                              query_tensors, response_tensors)
        
        # 🚨 CHECK KL DIVERGENCE BEFORE UPDATING
        current_kl = metrics.get('avg_kl_divergence', 0.0)
        if current_kl > self.max_kl_threshold:
            logger.warning(f"KL divergence {current_kl:.4f} exceeds threshold {self.max_kl_threshold:.4f}. Skipping PPO update.")
            # Return stats without updating
            stats = {
                "rewards/mean": sum(rewards) / len(rewards) if rewards else 0.0,
                "rewards/max": max(rewards) if rewards else 0.0,
                "rewards/min": min(rewards) if rewards else 0.0,
                "success_rate": sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0,
                "test_pass_rate": metrics.get('test_pass_rate', 0.0),
                "avg_test_ratio": metrics.get('avg_test_ratio', 0.0),
                "avg_quality_score": metrics.get('avg_quality_score', 0.0),
                "pass_at_1": metrics.get('perfect_solution_rate', 0.0),
                "syntax_errors": metrics.get('syntax_errors', 0),
                "total_tests": metrics.get('total_tests', 0),
                "passed_tests": metrics.get('total_passed_tests', 0),
                "avg_kl_divergence": metrics.get('avg_kl_divergence', 0.0),
                "avg_base_reward": metrics.get('avg_base_reward', 0.0),
                "policy_loss": 0.0,  # No update performed
                "value_loss": 0.0,
                "entropy_loss": 0.0,
                "ppo_epochs": 0,  # No epochs performed
                "update_skipped": True,
                "skip_reason": "kl_threshold_exceeded"
            }
            return stats
        
        # PPO Update Phase
        self.model.train()  # Set to train mode for gradient updates
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        update_count = 0
        
        # Perform multiple PPO epochs
        for ppo_epoch in range(self.ppo_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy_loss = 0.0
            
            for i in range(len(query_tensors)):
                if len(response_tensors[i]) == 0 or response_tensors[i].shape[-1] <= 1:
                    continue  # Skip failed generations
                
                # Prepare tensors
                query_tensor = query_tensors[i]
                response_tensor = response_tensors[i]
                full_sequence = torch.cat([query_tensor, response_tensor], dim=-1)
                attention_mask = torch.ones_like(full_sequence)
                
                # Compute current log probabilities and values
                new_log_probs = self.compute_log_probs(self.model, full_sequence, attention_mask)
                model_outputs = self.model(full_sequence, attention_mask=attention_mask)
                if hasattr(model_outputs, 'value') and model_outputs.value is not None:
                    new_values = model_outputs.value.squeeze(-1)
                elif isinstance(model_outputs, tuple) and len(model_outputs) >= 2:
                    # Handle tuple output format (logits, values, ...)
                    logits, values = model_outputs[0], model_outputs[1]
                    if values is not None:
                        new_values = values.squeeze(-1)
                    else:
                        new_values = torch.zeros((full_sequence.shape[0], full_sequence.shape[1]), device=device)
                else:
                    # Fallback: create zero values
                    new_values = torch.zeros((full_sequence.shape[0], full_sequence.shape[1]), device=device)
                
                # Get old log probabilities and values
                old_log_probs = old_log_probs_list[i].detach()
                old_values = old_values_list[i].detach()
                
                # Ensure tensors have the same length
                min_len = min(new_log_probs.shape[1], old_log_probs.shape[1])
                new_log_probs = new_log_probs[:, :min_len]
                old_log_probs = old_log_probs[:, :min_len]
                new_values = new_values[:, :min_len]
                old_values = old_values[:, :min_len]
                
                # Compute advantages and returns
                reward_tensor = torch.tensor([rewards[i]], device=device, dtype=torch.float32)
                advantages, returns = self.compute_advantages_and_returns(
                    [rewards[i]], new_values.mean(dim=1)  # Average over sequence length
                )
                
                # Expand advantages to match sequence length
                advantages_expanded = advantages[0].expand(new_log_probs.shape[1])
                returns_expanded = returns[0].expand(new_values.shape[1])
                
                # Compute losses with updated clip_range
                policy_loss = self.compute_policy_loss(
                    old_log_probs.sum(dim=1), 
                    new_log_probs.sum(dim=1), 
                    advantages_expanded.mean().unsqueeze(0),
                    clip_range=self.clip_range  # Use updated clip_range
                )
                
                value_loss = self.compute_value_loss(
                    new_values.mean(dim=1), 
                    returns_expanded.mean().unsqueeze(0)
                )
                
                # Compute entropy loss (for exploration)
                if hasattr(model_outputs, 'logits') and model_outputs.logits is not None:
                    logits = model_outputs.logits
                elif isinstance(model_outputs, tuple) and len(model_outputs) >= 1:
                    logits = model_outputs[0]  # First element should be logits
                else:
                    # Skip entropy loss if we can't get logits
                    entropy_loss = torch.tensor(0.0, device=device)
                    
                if 'logits' in locals():
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                    entropy_loss = -self.entropy_loss_coef * entropy
                
                # Total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
                
                # Backward pass and update
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping (更保守)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                # Accumulate losses
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                update_count += 1
            
            # Average losses for this epoch
            num_valid_samples = sum(1 for rt in response_tensors if len(rt) > 1)
            if num_valid_samples > 0:
                epoch_policy_loss /= num_valid_samples
                epoch_value_loss /= num_valid_samples
                epoch_entropy_loss /= num_valid_samples
            
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_entropy_loss += epoch_entropy_loss
        
        # Average losses across all PPO epochs
        avg_policy_loss = total_policy_loss / self.ppo_epochs
        avg_value_loss = total_value_loss / self.ppo_epochs
        avg_entropy_loss = total_entropy_loss / self.ppo_epochs
        
        # Return comprehensive statistics
        stats = {
            "rewards/mean": sum(rewards) / len(rewards) if rewards else 0.0,
            "rewards/max": max(rewards) if rewards else 0.0,
            "rewards/min": min(rewards) if rewards else 0.0,
            "success_rate": sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0,
            "test_pass_rate": metrics.get('test_pass_rate', 0.0),
            "avg_test_ratio": metrics.get('avg_test_ratio', 0.0),
            "avg_quality_score": metrics.get('avg_quality_score', 0.0),
            "pass_at_1": metrics.get('perfect_solution_rate', 0.0),
            "syntax_errors": metrics.get('syntax_errors', 0),
            "total_tests": metrics.get('total_tests', 0),
            "passed_tests": metrics.get('total_passed_tests', 0),
            "avg_kl_divergence": metrics.get('avg_kl_divergence', 0.0),
            "avg_base_reward": metrics.get('avg_base_reward', 0.0),
            # PPO-specific metrics
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "ppo_epochs": self.ppo_epochs,
            "update_count": update_count,
            "update_skipped": False,
        }
        
        return stats

    def train(self):
        """Run complete PPO training with gradient updates."""
        logger.info("Starting complete PPO training with gradient updates...")
        
        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_reward_function()
        
        # Initialize wandb if enabled
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
        
        # Training loop
        total_steps = 0
        best_success_rate = 0.0
        
        # Create batches from dataset
        batch_size = self.config.training.per_device_train_batch_size
        num_batches = len(self.dataset) // batch_size
        
        logger.info(f"Starting complete PPO training with {num_batches} batches per epoch")
        logger.info(f"Dataset size: {len(self.dataset)} samples")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"PPO epochs per step: {self.ppo_epochs}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Total samples per epoch: {num_batches * batch_size}")
        
        for epoch in range(self.config.training.num_train_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.training.num_train_epochs}")
            
            epoch_stats = []
            
            # Add progress bar for each epoch
            pbar = tqdm(
                range(num_batches), 
                desc=f"Epoch {epoch + 1}/{self.config.training.num_train_epochs}",
                unit="batch",
                ncols=140
            )
            
            for batch_idx in pbar:
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = self.dataset.select(range(start_idx, end_idx))
                
                # Perform complete PPO training step
                try:
                    step_stats = self.train_step_complete_ppo(batch_data)
                    epoch_stats.append(step_stats)
                    total_steps += 1
                    
                    # Update progress bar with current metrics
                    pbar.set_postfix({
                        'Reward': f"{step_stats.get('rewards/mean', 0.0):.3f}",
                        'Success': f"{step_stats.get('success_rate', 0.0):.2f}",
                        'Pass@1': f"{step_stats.get('pass_at_1', 0.0):.2f}",
                        'TestPass': f"{step_stats.get('test_pass_rate', 0.0):.2f}",
                        'KL': f"{step_stats.get('avg_kl_divergence', 0.0):.3f}",
                        'PolicyLoss': f"{step_stats.get('policy_loss', 0.0):.3f}",
                        'ValueLoss': f"{step_stats.get('value_loss', 0.0):.3f}"
                    })
                    
                    # Log statistics
                    if total_steps % self.config.training.logging_steps == 0:
                        logger.info(f"Step {total_steps}: {step_stats}")
                        
                        if self.config.use_wandb:
                            wandb.log(step_stats, step=total_steps)
                    
                    # Save checkpoint
                    if total_steps % self.config.training.save_steps == 0:
                        checkpoint_path = f"{self.config.training.output_dir}/checkpoint-{total_steps}"
                        save_model_and_tokenizer(
                            self.model.pretrained_model if hasattr(self.model, 'pretrained_model') else self.model, 
                            self.tokenizer, 
                            checkpoint_path,
                            push_to_hub=False
                        )
                        
                        # Save best model based on success rate
                        current_success_rate = step_stats.get("success_rate", 0.0)
                        if current_success_rate > best_success_rate:
                            best_success_rate = current_success_rate
                            best_model_path = f"{self.config.training.output_dir}/best_model"
                            save_model_and_tokenizer(
                                self.model.pretrained_model if hasattr(self.model, 'pretrained_model') else self.model, 
                                self.tokenizer, 
                                best_model_path,
                                push_to_hub=False
                            )
                            logger.info(f"New best model saved with success rate: {best_success_rate:.4f}")
                
                except Exception as e:
                    logger.error(f"Error in training step {total_steps}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Close progress bar and log epoch summary
            pbar.close()
            
            # Calculate epoch summary statistics
            if epoch_stats:
                avg_reward = sum(s.get('rewards/mean', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_success_rate = sum(s.get('success_rate', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_test_pass_rate = sum(s.get('test_pass_rate', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_pass_at_1 = sum(s.get('pass_at_1', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_kl = sum(s.get('avg_kl_divergence', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_base_reward = sum(s.get('avg_base_reward', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_policy_loss = sum(s.get('policy_loss', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_value_loss = sum(s.get('value_loss', 0.0) for s in epoch_stats) / len(epoch_stats)
                avg_entropy_loss = sum(s.get('entropy_loss', 0.0) for s in epoch_stats) / len(epoch_stats)
                total_syntax_errors = sum(s.get('syntax_errors', 0) for s in epoch_stats)
                skipped_updates = sum(1 for s in epoch_stats if s.get('update_skipped', False))
                
                logger.info(f"Epoch {epoch + 1} Summary:")
                logger.info(f"  Average Final Reward: {avg_reward:.4f}")
                logger.info(f"  Average Base Reward: {avg_base_reward:.4f}")
                logger.info(f"  Average KL Divergence: {avg_kl:.4f}")
                logger.info(f"  Average Success Rate: {avg_success_rate:.4f}")
                logger.info(f"  Average Pass@1: {avg_pass_at_1:.4f}")
                logger.info(f"  Average Test Pass Rate: {avg_test_pass_rate:.4f}")
                logger.info(f"  Average Policy Loss: {avg_policy_loss:.4f}")
                logger.info(f"  Average Value Loss: {avg_value_loss:.4f}")
                logger.info(f"  Average Entropy Loss: {avg_entropy_loss:.4f}")
                logger.info(f"  Total Syntax Errors: {total_syntax_errors}")
                logger.info(f"  Skipped Updates: {skipped_updates}/{len(epoch_stats)}")
                
                # 🚨 EARLY STOPPING CHECK
                # 使用测试通过率作为主要指标，奖励作为辅助指标
                current_metric = avg_test_pass_rate
                
                if current_metric > self.best_test_pass_rate:
                    # 性能提升，重置patience计数器
                    self.best_test_pass_rate = current_metric
                    self.best_reward = avg_reward
                    self.patience_counter = 0
                    
                    # 保存最佳模型
                    best_model_path = f"{self.config.training.output_dir}/best_epoch_model"
                    save_model_and_tokenizer(
                        self.model.pretrained_model if hasattr(self.model, 'pretrained_model') else self.model, 
                        self.tokenizer, 
                        best_model_path,
                        push_to_hub=False
                    )
                    logger.info(f"🎯 NEW BEST EPOCH! Test pass rate: {current_metric:.4f}, Reward: {avg_reward:.4f}")
                    logger.info(f"💾 Best model saved to: {best_model_path}")
                    
                else:
                    # 性能下降或持平，增加patience计数器
                    self.patience_counter += 1
                    performance_drop = self.best_test_pass_rate - current_metric
                    logger.info(f"⚠️  Performance drop detected: {performance_drop:.4f} (patience: {self.patience_counter}/{self.patience})")
                    
                    if self.patience_counter >= self.patience:
                        logger.warning(f"🛑 EARLY STOPPING: No improvement for {self.patience} epochs")
                        logger.info(f"📊 Best test pass rate: {self.best_test_pass_rate:.4f}")
                        logger.info(f"📊 Current test pass rate: {current_metric:.4f}")
                        self.early_stop = True
                
                # 额外的KL散度安全检查
                if avg_kl > 1.0:  # 如果KL散度过大，强制停止
                    logger.warning(f"🚨 EMERGENCY STOP: KL divergence {avg_kl:.4f} is too high!")
                    self.early_stop = True
                
                # Log epoch summary to wandb
                if self.config.use_wandb:
                    wandb.log({
                        f"epoch_{epoch + 1}/avg_reward": avg_reward,
                        f"epoch_{epoch + 1}/avg_base_reward": avg_base_reward,
                        f"epoch_{epoch + 1}/avg_kl_divergence": avg_kl,
                        f"epoch_{epoch + 1}/avg_success_rate": avg_success_rate,
                        f"epoch_{epoch + 1}/avg_pass_at_1": avg_pass_at_1,
                        f"epoch_{epoch + 1}/avg_test_pass_rate": avg_test_pass_rate,
                        f"epoch_{epoch + 1}/avg_policy_loss": avg_policy_loss,
                        f"epoch_{epoch + 1}/avg_value_loss": avg_value_loss,
                        f"epoch_{epoch + 1}/avg_entropy_loss": avg_entropy_loss,
                        f"epoch_{epoch + 1}/syntax_errors": total_syntax_errors,
                        f"epoch_{epoch + 1}/skipped_updates": skipped_updates,
                        f"epoch_{epoch + 1}/patience_counter": self.patience_counter,
                        f"epoch_{epoch + 1}/best_test_pass_rate": self.best_test_pass_rate,
                    }, step=total_steps)
            
            # 检查是否需要早停
            if self.early_stop:
                logger.info(f"🏁 Training stopped early at epoch {epoch + 1}")
                break
        
        # Save final model
        logger.info("Training completed. Saving final model...")
        final_model_path = f"{self.config.training.output_dir}/final_model"
        save_model_and_tokenizer(
            self.model.pretrained_model if hasattr(self.model, 'pretrained_model') else self.model, 
            self.tokenizer, 
            final_model_path,
            push_to_hub=self.config.hf.push_to_hub,
            hub_model_id=self.config.hf.hub_model_id,
            hub_token=self.config.hf.hub_token
        )
        
        if self.config.use_wandb:
            wandb.finish()
        
        logger.info("Complete PPO training completed successfully!")


def main():
    """Main function to run PPO training."""
    # Load configuration
    from config import load_config_from_env
    config = load_config_from_env()
    config.method = "ppo"
    
    # Create trainer and run training
    trainer = CodePPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 