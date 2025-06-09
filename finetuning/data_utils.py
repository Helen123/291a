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


class HumanEvalDataProcessor:
    """Process HumanEval dataset for training with reinforcement learning."""
    
    # Stop words from bigcode-evaluation-harness HumanEval task (exact match)
    STOP_WORDS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"]
    
    def __init__(self, dataset_name: str = "openai_humaneval", split: str = "test", strip_prompt: bool = True):
        """
        Initialize HumanEval data processor.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split to use (HumanEval only has test split)
            strip_prompt: Whether to strip the prompt (default True, like bigcode-evaluation-harness)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.strip_prompt = strip_prompt
        self.dataset = None
        # Set stop_words as instance attribute like bigcode-evaluation-harness
        self.stop_words = self.STOP_WORDS
        
    def load_dataset(self) -> Dataset:
        """Load HumanEval dataset."""
        if self.dataset is None:
            # Load full dataset like bigcode-evaluation-harness
            self.dataset = load_dataset(self.dataset_name)
            # Validate dataset size
            assert len(self.dataset[self.split]) == 164, \
                "HumanEval dataset should have 164 problems"
        return self.dataset[self.split]  # Return the specific split for compatibility
    
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset[self.split]
    
    def create_prompt(self, doc: Dict[str, Any]) -> str:
        """
        Create prompt from HumanEval document.
        Matches bigcode-evaluation-harness implementation exactly.
        
        Args:
            doc: HumanEval document
            
        Returns:
            Formatted prompt
        """
        if self.strip_prompt:
            return doc["prompt"].strip()
        else:
            return doc["prompt"]
    
    def get_prompt(self, doc: Dict[str, Any]) -> str:
        """
        Builds the prompt for the LM to generate from.
        Exact alias for create_prompt to match bigcode-evaluation-harness interface.
        """
        return self.create_prompt(doc)
    
    def _stop_at_stop_token(self, generation: str, stop_words: List[str]) -> str:
        """
        Stop generation at first occurrence of any stop word.
        
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
    
    def postprocess_generation(self, generation: str, idx: int) -> str:
        """
        Postprocess generation for HumanEval.
        Matches bigcode-evaluation-harness implementation exactly.
        
        Args:
            generation: Raw generated text from model
            idx: Index of the document in the dataset
            
        Returns:
            Postprocessed generation
        """
        # Get the prompt for this specific example (exact bigcode format)
        prompt = self.get_prompt(self.dataset[self.split][idx])
        
        # Remove prompt from generation (exact bigcode format)
        generation_only = generation[len(prompt):]
        
        # Apply stop word filtering and return (exact bigcode format)
        return prompt + self._stop_at_stop_token(generation_only, self.stop_words)
    
    def extract_function_signature(self, code: str) -> Optional[str]:
        """Extract function signature from code."""
        lines = code.strip().split('\n')
        for line in lines:
            if line.strip().startswith('def '):
                return line.strip()
        return None
    
    def prepare_training_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Prepare HumanEval training data.
        
        Args:
            max_samples: Maximum number of samples to process
            
        Returns:
            List of training examples
        """
        dataset = self.load_dataset()  # This returns the specific split
        training_data = []
        
        max_samples = max_samples or len(dataset)
        
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
            
            prompt = self.create_prompt(example)
            
            training_data.append({
                "prompt": prompt,
                "test_cases": example["test"],
                "reference_code": example["canonical_solution"],
                "task_id": example["task_id"],
                "description": example.get("description", ""),
                "entry_point": example["entry_point"]
            })
        
        return training_data

    def get_reference(self, doc: Dict[str, Any]) -> str:
        """
        Build the reference solution for the doc (sample from the test dataset).
        Matches bigcode-evaluation-harness implementation exactly.
        
        Args:
            doc: HumanEval document
            
        Returns:
            Reference test code with entry point call
        """
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    def postprocess_generation_with_prompt(self, generation: str, prompt: str) -> str:
        """
        Backward compatibility method for existing training code.
        
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
        cleaned_generation = self._stop_at_stop_token(generation_only, self.stop_words)
        
        # For HumanEval, ensure proper formatting between docstring and code
        # The prompt ends with '"""' so we need a newline before the actual code
        if not cleaned_generation.startswith('\n') and cleaned_generation.strip():
            cleaned_generation = '\n' + cleaned_generation
        
        # Return prompt + cleaned generation
        return prompt + cleaned_generation


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
    """Compute rewards for generated code based on test pass ratio and quality."""
    
    def __init__(self, timeout: float = 10.0, max_reward: float = 1.0, 
                 min_reward: float = 0.0, syntax_penalty: float = -0.2,
                 quality_bonus: float = 0.1, dataset_type: str = "mbpp"):
        """
        Initialize reward function.
        
        Args:
            timeout: Code execution timeout
            max_reward: Maximum reward value
            min_reward: Minimum reward value  
            syntax_penalty: Penalty for syntax errors
            quality_bonus: Bonus for code quality
            dataset_type: Type of dataset ("mbpp" or "humaneval")
        """
        self.timeout = timeout
        self.max_reward = max_reward
        self.min_reward = min_reward
        self.syntax_penalty = syntax_penalty
        self.quality_bonus = quality_bonus
        self.dataset_type = dataset_type.lower()
        
        # Initialize appropriate processor
        if self.dataset_type == "mbpp":
            self.processor = MBPPDataProcessor()
        elif self.dataset_type == "humaneval":
            self.processor = HumanEvalDataProcessor()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        print(f"RewardFunction initialized for {self.dataset_type.upper()} dataset")
    
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
        # For HumanEval, we need to add the function call like bigcode-evaluation-harness
        if self.dataset_type == "humaneval":
            # Extract function name from code
            import re
            func_match = re.search(r'def\s+(\w+)\s*\(', code)
            if func_match:
                func_name = func_match.group(1)
                # Add the check call like bigcode does: test_func + "\n" + f"check({func_name})"
                test_with_call = test_cases + f"\ncheck({func_name})"
            else:
                # Fallback if we can't extract function name
                test_with_call = test_cases
        else:
            test_with_call = test_cases
        
        # Use bigcode-style evaluation - run all tests together
        result = check_code_correctness(code, test_with_call, self.timeout)
        
        if result["passed"]:
            return 1.0, 1, 1
        else:
            # Conservative reward strategy for HumanEval - reduce misleading partial rewards
            # Only provide small syntax and structure rewards when completely failing
            partial_reward = 0.0
            
            # 1. Syntax correctness reward (more conservative)
            try:
                compile(code, '<string>', 'exec')
                partial_reward += 0.1  # Significantly reduce syntax reward
            except SyntaxError:
                return -0.1, 0, 1  # Give slight negative reward for syntax errors
            
            # 2. Function definition existence check (more conservative)
            if 'def ' in code and 'return ' in code:
                partial_reward += 0.1  # Only give reward when both def and return are present
            
            # 3. Additional penalty for avoiding common error patterns
            if 'pass' == code.strip().split('\n')[-1].strip():
                partial_reward -= 0.05  # Reduce reward if only pass is present
            
            if partial_reward < 0.1:
                partial_reward = 0.0  # Set to 0 directly if partial reward is too low
            
            return partial_reward, 0, 1
    
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
        if self.dataset_type == "humaneval":
            postprocessed_generation = self.processor.postprocess_generation_with_prompt(generated_text, prompt)
            # For HumanEval, use the complete function (prompt + generation) for evaluation
            # This matches bigcode-evaluation-harness behavior: test_program = candidate + "\n" + test_case
            code_for_evaluation = postprocessed_generation
            code_for_quality = postprocessed_generation[len(prompt):] if postprocessed_generation.startswith(prompt) else postprocessed_generation
        else:
            postprocessed_generation = self.processor.postprocess_generation(generated_text, prompt)
            # For MBPP, extract only the generated code part (remove prompt)
            if postprocessed_generation.startswith(prompt):
                code_for_evaluation = postprocessed_generation[len(prompt):]
                code_for_quality = code_for_evaluation
            else:
                code_for_evaluation = postprocessed_generation
                code_for_quality = postprocessed_generation
        
        # Check for syntax errors first (only check the generated part for quality assessment)
        try:
            if self.dataset_type == "humaneval":
                # For HumanEval, check the complete function
                compile(code_for_evaluation, '<string>', 'exec')
            else:
                # For MBPP, check the generated code part
                compile(code_for_evaluation, '<string>', 'exec')
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
                'postprocessed_code': code_for_quality
            }
        
        # Compute test pass ratio using the SAME method as evaluation
        pass_ratio, passed_tests, total_tests = self.compute_test_pass_ratio(code_for_evaluation, test_cases)
        
        # Base reward based on test pass ratio
        base_reward = self.min_reward + (self.max_reward - self.min_reward) * pass_ratio
        
        # Quality bonus (always use the generated part for quality assessment)
        quality_score = self.evaluate_code_quality(code_for_quality)
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
            'postprocessed_code': code_for_quality
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
                                     test_cases: List[str], max_workers: int = 8) -> Tuple[List[float], List[Dict]]:
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
                            test_cases: List[str], max_workers: int = 8) -> List[float]:
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


def create_humaneval_dataset_for_rl(split: str = "test", max_samples: Optional[int] = None) -> Dataset:
    """
    Create HumanEval dataset formatted for RL training.
    
    Args:
        split: Dataset split to use (only 'test' available for HumanEval)
        max_samples: Maximum number of samples
        
    Returns:
        Processed dataset
    """
    processor = HumanEvalDataProcessor(split=split)
    training_data = processor.prepare_training_data(max_samples)
    
    # Convert to HuggingFace dataset
    dataset_dict = {
        "prompt": [item["prompt"] for item in training_data],
        "test_cases": [item["test_cases"] for item in training_data],
        "reference_code": [item["reference_code"] for item in training_data],
        "task_id": [item["task_id"] for item in training_data],
        "description": [item["description"] for item in training_data],
        "entry_point": [item["entry_point"] for item in training_data]
    }
    
    return Dataset.from_dict(dataset_dict)


def create_dataset_for_rl(dataset_name: str = "mbpp", split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """
    Create dataset formatted for RL training - supports both MBPP and HumanEval.
    
    Args:
        dataset_name: Name of the dataset ("mbpp" or "humaneval")
        split: Dataset split to use
        max_samples: Maximum number of samples
        
    Returns:
        Processed dataset
    """
    if dataset_name.lower() == "mbpp":
        return create_mbpp_dataset_for_rl(split=split, max_samples=max_samples)
    elif dataset_name.lower() == "humaneval":
        # HumanEval only has test split, but we use it for training
        return create_humaneval_dataset_for_rl(split="test", max_samples=max_samples)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'mbpp' or 'humaneval'.") 