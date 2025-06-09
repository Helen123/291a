#!/usr/bin/env python3
"""
Merge LoRA adapter to base model, generate complete model for evaluation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from pathlib import Path
import json

def merge_and_save_model(adapter_model_id: str, output_dir: str, base_model_name: str = None, push_to_hub: bool = False, hub_model_id: str = None, device: str = "auto"):
    """
    Merge adapter to base model and save complete model
    
    Args:
        adapter_model_id: HuggingFace adapter model ID or local path
        output_dir: Output directory  
        base_model_name: Base model name (optional, will auto-detect from adapter config)
        push_to_hub: Whether to upload to HuggingFace
        hub_model_id: HuggingFace model ID
        device: Device selection ("auto", "cpu", "cuda:0", "cuda:3", etc.)
    """
    print("🔧 Loading base model...")
    print(f"🎯 Using device: {device}")
    
    # First load adapter's tokenizer to get correct vocabulary size
    print(f"🔍 Checking adapter tokenizer: {adapter_model_id}")
    adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_model_id)
    adapter_vocab_size = len(adapter_tokenizer)
    print(f"📊 Adapter vocabulary size: {adapter_vocab_size}")
    
    # Auto-detect base model name
    if base_model_name is None:
        try:
            # Try to read base model info from adapter config
            config_path = Path(adapter_model_id) / "adapter_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    adapter_config = json.load(f)
                    if 'base_model_name_or_path' in adapter_config:
                        base_model_name = adapter_config['base_model_name_or_path']
                        print(f"🎯 Detected base model from adapter config: {base_model_name}")
            else:
                # Default to CodeLlama
                base_model_name = "codellama/CodeLlama-7b-Python-hf"
                print(f"⚠️  Adapter config not found, using default base model: {base_model_name}")
        except Exception as e:
            base_model_name = "codellama/CodeLlama-7b-Python-hf"
            print(f"⚠️  Failed to detect base model, using default: {base_model_name}")
    else:
        print(f"🤖 Using specified base model: {base_model_name}")
    
    # Set device mapping
    if device == "cpu":
        device_map = {"": "cpu"}
        torch_dtype = torch.float32  # Use float32 for CPU
        print("💻 Using CPU for merge")
    elif device == "auto":
        device_map = "auto"
        torch_dtype = torch.float16
        print("🚀 Using automatic device mapping")
    else:
        # Specify specific GPU
        device_map = {"": device}
        torch_dtype = torch.float16
        print(f"🎯 Using specified GPU: {device}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # Load base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_vocab_size = len(base_tokenizer)
    print(f"📊 Base model vocabulary size: {base_vocab_size}")
    
    # If vocabulary sizes don't match, need to adjust base model
    if adapter_vocab_size != base_vocab_size:
        print(f"⚠️  Vocabulary size mismatch! Resizing base model vocabulary: {base_vocab_size} -> {adapter_vocab_size}")
        base_model.resize_token_embeddings(adapter_vocab_size)
        print("✅ Base model vocabulary resized")
    
    print(f"🔗 Loading adapter: {adapter_model_id}")
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, adapter_model_id)
    
    print("🔄 Merging adapter to base model...")
    
    # Merge adapter weights to base model
    merged_model = model.merge_and_unload()
    
    print(f"💾 Saving merged model to: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save merged model
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    
    # Use adapter's tokenizer (includes correct vocabulary)
    adapter_tokenizer.save_pretrained(output_dir)
    
    print("✅ Merging completed!")
    
    # Verify merge results
    print("🔍 Verifying merge results...")
    try:
        # Try to load merged model (use CPU to avoid GPU memory issues)
        test_model = AutoModelForCausalLM.from_pretrained(
            output_dir, 
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="cpu"  # Use CPU for verification
        )
        test_tokenizer = AutoTokenizer.from_pretrained(output_dir)
        print(f"✅ Verification successful! Merged model vocabulary size: {len(test_tokenizer)}")
        
        # Clean up memory
        del test_model
        if device != "cpu":
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
    
    # Upload to HuggingFace Hub
    if push_to_hub and hub_model_id:
        print(f"📤 Uploading to HuggingFace Hub: {hub_model_id}")
        merged_model.push_to_hub(hub_model_id)
        adapter_tokenizer.push_to_hub(hub_model_id)
        print("✅ Upload completed!")
        return hub_model_id
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter to base model")
    parser.add_argument("--adapter_model_id", type=str, 
                       default="yuexishen/codellama-7b-mbpp-ppo-qlora",
                       help="HuggingFace adapter model ID or local path")
    parser.add_argument("--output_dir", type=str, 
                       default="./merged_model",
                       help="Output directory")
    parser.add_argument("--base_model_name", type=str,
                       help="Base model name (optional, auto-detect from adapter config)")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Upload to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str,
                       help="HuggingFace Hub model ID")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device selection: auto, cpu, cuda:0, cuda:3, etc.")
    
    args = parser.parse_args()
    
    print("🚀 Starting merge process...")
    print(f"📂 Adapter model: {args.adapter_model_id}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Device: {args.device}")
    if args.base_model_name:
        print(f"🤖 Base model: {args.base_model_name}")
    else:
        print("🤖 Base model: auto-detect")
    
    if args.push_to_hub and not args.hub_model_id:
        print("❌ Error: Hub upload requires hub_model_id")
        return
    
    try:
        merged_model_path = merge_and_save_model(
            adapter_model_id=args.adapter_model_id,
            output_dir=args.output_dir,
            base_model_name=args.base_model_name,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            device=args.device
        )
        
        print("🎯 Merge completed! Now ready for evaluation:")
        print(f"Model path: {merged_model_path}")
        
        # Output evaluation command examples
        print("\n📋 BigCode evaluation command:")
        if args.push_to_hub and args.hub_model_id:
            model_name = args.hub_model_id
        else:
            model_name = args.output_dir
            
        print(f"""
cd ../bigcode-evaluation-harness
accelerate launch main.py \\
  --model {model_name} \\
  --max_length_generation 512 \\
  --tasks mbpp \\
  --temperature 0.1 \\
  --n_samples 15 \\
  --batch_size 10 \\
  --allow_code_execution
        """)
        
        print("\n🔬 Or use our local evaluation script:")
        print(f"python evaluate.py {model_name} --num_samples 15 --temperature 0.1")
        
    except Exception as e:
        print(f"❌ Merge failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 