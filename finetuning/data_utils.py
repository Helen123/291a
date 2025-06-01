import os
import re
import tempfile
import subprocess
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Dict, List, Tuple, Any, Optional
from functools import partial
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm


class CodeExecutionError(Exception):
    """Exception for code execution errors."""
    pass


def execute_code_with_timeout(code: str, timeout: float = 10.0) -> Tuple[bool, str]:
    """
    Execute Python code with timeout and return success status and output.
    
    Args:
        code: Python code to execute
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute the code with timeout
            result = subprocess.run(
                ["python", temp_file],
                timeout=timeout,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Execution timeout"
        except Exception as e:
            return False, str(e)
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
                
    except Exception as e:
        return False, str(e)


def check_code_correctness(code: str, test_cases: str, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Check if code passes test cases.
    
    Args:
        code: Generated code
        test_cases: Test cases to run
        timeout: Execution timeout
        
    Returns:
        Dictionary with results
    """
    # Combine code with test cases
    full_program = f"{code}\n\n{test_cases}"
    
    # Execute and check
    success, output = execute_code_with_timeout(full_program, timeout)
    
    return {
        "passed": success,
        "output": output,
        "code": code,
        "test_cases": test_cases
    }


class MBPPDataProcessor:
    """Process MBPP dataset for training with reinforcement learning."""
    
    # Stop words from bigcode-evaluation-harness MBPP task
    STOP_WORDS = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]
    
    def __init__(self, dataset_name: str = "mbpp", split: str = "train"):
        """
        Initialize MBPP data processor.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split to use
        """
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None
        
    def load_dataset(self) -> Dataset:
        """Load MBPP dataset."""
        if self.dataset is None:
            self.dataset = load_dataset(self.dataset_name)[self.split]
            # Validate dataset size for test split
            if self.split == "test":
                assert len(self.dataset) == 500, \
                    "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return self.dataset
    
    def create_prompt(self, doc: Dict[str, Any]) -> str:
        """
        Create prompt from MBPP document following bigcode-evaluation-harness format.
        MBPP prompt is built following InCoder (Fried et al.) approach.
        
        Args:
            doc: MBPP document
            
        Returns:
            Formatted prompt identical to bigcode-evaluation-harness
        """
        description = doc["text"]
        test_example = doc["test_list"][0]
        prompt = f'"""\n{description}\n{test_example}\n"""\n'
        return prompt
    
    def _stop_at_stop_token(self, generation: str, stop_words: List[str]) -> str:
        """
        Stop generation at first occurrence of any stop word.
        Identical to bigcode-evaluation-harness implementation.
        
        Args:
            generation: Generated text
            stop_words: List of stop words
            
        Returns:
            Truncated generation
        """
        stop_indices = []
        for stop_word in stop_words:
            idx = generation.find(stop_word)
            if idx != -1:
                stop_indices.append(idx)
        
        if stop_indices:
            generation = generation[:min(stop_indices)]
        
        return generation
    
    def postprocess_generation(self, generation: str, prompt: str) -> str:
        """
        Postprocess generation exactly like bigcode-evaluation-harness.
        
        Args:
            generation: Raw generated text from model
            prompt: Original prompt
            
        Returns:
            Postprocessed generation
        """
        # Remove prompt from generation
        if generation.startswith(prompt):
            generation_only = generation[len(prompt):]
        else:
            generation_only = generation
        
        # Apply stop word filtering
        cleaned_generation = self._stop_at_stop_token(generation_only, self.STOP_WORDS)
        
        # Return prompt + cleaned generation (as per bigcode format)
        return prompt + cleaned_generation
    
    def clean_generated_code(self, generated_code: str, prompt: str) -> str:
        """
        Clean generated code by removing prompt and stopping at stop tokens.
        This is the legacy method, use postprocess_generation for bigcode compatibility.
        
        Args:
            generated_code: Raw generated code
            prompt: Original prompt
            
        Returns:
            Cleaned code
        """
        return self.postprocess_generation(generated_code, prompt)
    
    def extract_function_signature(self, code: str) -> Optional[str]:
        """Extract function signature from code."""
        lines = code.strip().split('\n')
        for line in lines:
            if line.strip().startswith('def '):
                return line.strip()
        return None
    
    def prepare_training_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Prepare training data for RL fine-tuning.
        
        Args:
            max_samples: Maximum number of samples to use
            
        Returns:
            List of training examples
        """
        dataset = self.load_dataset()
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        training_data = []
        
        for example in tqdm(dataset, desc="Preparing training data"):
            prompt = self.create_prompt(example)
            test_cases = "\n".join(example["test_list"])
            
            training_example = {
                "prompt": prompt,
                "test_cases": test_cases,
                "reference_code": example["code"],
                "task_id": example["task_id"],
                "description": example["text"]
            }
            
            training_data.append(training_example)
        
        return training_data


class RewardFunction:
    """Reward function based on test case pass ratio for code generation."""
    
    def __init__(self, timeout: float = 10.0, max_reward: float = 1.0, 
                 min_reward: float = 0.0, syntax_penalty: float = -0.2,
                 quality_bonus: float = 0.1):
        """
        Initialize reward function.
        
        Args:
            timeout: Code execution timeout
            max_reward: Maximum reward for passing all tests
            min_reward: Minimum reward for passing no tests
            syntax_penalty: Penalty for syntax errors
            quality_bonus: Bonus multiplier for code quality
        """
        self.timeout = timeout
        self.max_reward = max_reward
        self.min_reward = min_reward
        self.syntax_penalty = syntax_penalty
        self.quality_bonus = quality_bonus
        self.processor = MBPPDataProcessor()
    
    def parse_test_cases(self, test_cases: str) -> List[str]:
        """
        Parse test cases into individual test statements.
        
        Args:
            test_cases: Raw test cases string
            
        Returns:
            List of individual test statements
        """
        test_lines = []
        for line in test_cases.split('\n'):
            line = line.strip()
            if line and (line.startswith('assert') or line.startswith('print(')):
                test_lines.append(line)
        return test_lines
    
    def run_individual_test(self, code: str, test_statement: str) -> bool:
        """
        Run a single test statement against the code.
        
        Args:
            code: Generated code
            test_statement: Single test statement
            
        Returns:
            True if test passes, False otherwise
        """
        try:
            # Create complete test code
            full_code = f"{code}\n{test_statement}"
            success, output = execute_code_with_timeout(full_code, self.timeout)
            return success
        except Exception:
            return False
    
    def compute_test_pass_ratio(self, code: str, test_cases: str) -> Tuple[float, int, int]:
        """
        Compute the ratio of tests that pass.
        This is the MAIN method used during training for reward calculation.
        
        Args:
            code: Generated code
            test_cases: Test cases string
            
        Returns:
            Tuple of (pass_ratio, passed_count, total_count)
        """
        # Use bigcode-style evaluation - run all tests together
        result = check_code_correctness(code, test_cases, self.timeout)
        
        if result["passed"]:
            return 1.0, 1, 1
        else:
            # If all tests together fail, check individual tests for partial credit
            test_statements = self.parse_test_cases(test_cases)
            
            if not test_statements:
                return 0.0, 0, 1
            
            # Run each test individually for better debugging/partial rewards
            passed_tests = 0
            total_tests = len(test_statements)
            
            for test_stmt in test_statements:
                if self.run_individual_test(code, test_stmt):
                    passed_tests += 1
            
            pass_ratio = passed_tests / total_tests if total_tests > 0 else 0.0
            return pass_ratio, passed_tests, total_tests
    
    def evaluate_code_quality(self, code: str) -> float:
        """
        Simple code quality evaluation.
        
        Args:
            code: Code to evaluate
            
        Returns:
            Quality score between 0 and 1
        """
        if not code.strip():
            return 0.0
        
        score = 0.0
        
        # Has function definition
        if 'def ' in code:
            score += 0.3
        
        # Has return statement
        if 'return ' in code:
            score += 0.3
        
        # Reasonable length (not too short or too long)
        code_length = len(code.strip())
        if 20 <= code_length <= 300:
            score += 0.2
        elif 10 <= code_length <= 500:
            score += 0.1
        
        # Basic structure (indentation)
        if any(line.startswith('    ') or line.startswith('\t') for line in code.split('\n')):
            score += 0.2
        
        return min(score, 1.0)
    
    def compute_reward(self, generated_text: str, prompt: str, test_cases: str) -> Dict[str, float]:
        """
        Compute reward based on test pass ratio and code quality.
        Uses bigcode-style postprocessing before evaluation.
        
        Args:
            generated_text: Raw generated text from model
            prompt: Original prompt
            test_cases: Test cases string
            
        Returns:
            Dictionary with reward details
        """
        # Apply bigcode-style postprocessing
        postprocessed_generation = self.processor.postprocess_generation(generated_text, prompt)
        
        # Extract only the generated code part (remove prompt)
        if postprocessed_generation.startswith(prompt):
            code = postprocessed_generation[len(prompt):]
        else:
            code = postprocessed_generation
        
        # Check for syntax errors first
        try:
            compile(code, '<string>', 'exec')
            has_syntax_error = False
        except SyntaxError:
            has_syntax_error = True
        
        if has_syntax_error:
            return {
                'total_reward': self.syntax_penalty,
                'test_ratio': 0.0,
                'passed_tests': 0,
                'total_tests': 0,
                'quality_score': 0.0,
                'base_reward': self.syntax_penalty,
                'quality_bonus': 0.0,
                'has_syntax_error': True,
                'postprocessed_code': code
            }
        
        # Compute test pass ratio using the SAME method as evaluation
        pass_ratio, passed_tests, total_tests = self.compute_test_pass_ratio(code, test_cases)
        
        # Base reward based on test pass ratio
        base_reward = self.min_reward + (self.max_reward - self.min_reward) * pass_ratio
        
        # Quality bonus
        quality_score = self.evaluate_code_quality(code)
        quality_bonus_value = quality_score * self.quality_bonus
        
        # Total reward
        total_reward = base_reward + quality_bonus_value
        
        return {
            'total_reward': total_reward,
            'test_ratio': pass_ratio,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'quality_score': quality_score,
            'base_reward': base_reward,
            'quality_bonus': quality_bonus_value,
            'has_syntax_error': False,
            'postprocessed_code': code
        }
    
    def __call__(self, generated_codes: List[str], prompts: List[str], 
                 test_cases: List[str]) -> List[float]:
        """
        Compute rewards for generated codes.
        
        Args:
            generated_codes: List of generated code strings
            prompts: List of prompts
            test_cases: List of test cases
            
        Returns:
            List of reward values
        """
        rewards = []
        
        for code, prompt, tests in zip(generated_codes, prompts, test_cases):
            # Compute reward with bigcode-style processing
            reward_details = self.compute_reward(code, prompt, tests)
            rewards.append(reward_details['total_reward'])
        
        return rewards
    
    def compute_batch_rewards_detailed(self, generated_codes: List[str], prompts: List[str], 
                                     test_cases: List[str], max_workers: int = 4) -> Tuple[List[float], List[Dict]]:
        """
        Compute detailed rewards for a batch of codes.
        
        Args:
            generated_codes: List of generated code strings
            prompts: List of prompts
            test_cases: List of test cases
            max_workers: Maximum number of worker threads
            
        Returns:
            Tuple of (reward_values, reward_details)
        """
        # Prepare work items
        work_items = []
        for i, (code, prompt, tests) in enumerate(zip(generated_codes, prompts, test_cases)):
            work_items.append((i, code, prompt, tests))
        
        # Execute in parallel
        rewards = [0.0] * len(generated_codes)
        details = [{}] * len(generated_codes)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for idx, code, prompt, tests in work_items:
                future = executor.submit(self.compute_reward, code, prompt, tests)
                future_to_index[future] = idx
            
            # Collect results
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    reward_details = future.result()
                    rewards[idx] = reward_details['total_reward']
                    details[idx] = reward_details
                except Exception:
                    rewards[idx] = self.syntax_penalty
                    details[idx] = {
                        'total_reward': self.syntax_penalty,
                        'test_ratio': 0.0,
                        'passed_tests': 0,
                        'total_tests': 0,
                        'has_syntax_error': True,
                        'error': 'Execution failed'
                    }
        
        return rewards, details
    
    def compute_batch_rewards(self, generated_codes: List[str], prompts: List[str], 
                            test_cases: List[str], max_workers: int = 4) -> List[float]:
        """
        Compute rewards for a batch of codes using parallel execution.
        
        Args:
            generated_codes: List of generated code strings
            prompts: List of prompts
            test_cases: List of test cases
            max_workers: Maximum number of worker threads
            
        Returns:
            List of rewards
        """
        rewards, _ = self.compute_batch_rewards_detailed(generated_codes, prompts, test_cases, max_workers)
        return rewards


def create_mbpp_dataset_for_rl(split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """
    Create MBPP dataset formatted for RL training.
    
    Args:
        split: Dataset split to use
        max_samples: Maximum number of samples
        
    Returns:
        Processed dataset
    """
    processor = MBPPDataProcessor(split=split)
    training_data = processor.prepare_training_data(max_samples)
    
    # Convert to HuggingFace dataset
    dataset_dict = {
        "prompt": [item["prompt"] for item in training_data],
        "test_cases": [item["test_cases"] for item in training_data],
        "reference_code": [item["reference_code"] for item in training_data],
        "task_id": [item["task_id"] for item in training_data],
        "description": [item["description"] for item in training_data]
    }
    
    return Dataset.from_dict(dataset_dict) 