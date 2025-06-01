#!/usr/bin/env python3
"""
合并LoRA适配器到基础模型，生成完整模型用于评估
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from pathlib import Path

def merge_and_save_model(adapter_model_id: str, output_dir: str, push_to_hub: bool = False, hub_model_id: str = None):
    """
    合并adapter到基础模型并保存完整模型
    
    Args:
        adapter_model_id: HuggingFace adapter模型ID
        output_dir: 输出目录
        push_to_hub: 是否上传到HuggingFace
        hub_model_id: HuggingFace模型ID
    """
    
    print("🔧 正在加载基础模型...")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-Python-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
    
    print(f"🔗 正在加载adapter: {adapter_model_id}")
    
    # 加载PEFT模型
    model = PeftModel.from_pretrained(base_model, adapter_model_id)
    
    print("🔄 正在合并adapter到基础模型...")
    
    # 合并adapter权重到基础模型
    merged_model = model.merge_and_unload()
    
    print(f"💾 正在保存合并后的模型到: {output_dir}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存合并后的模型
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    print("✅ 模型合并完成!")
    
    # 上传到HuggingFace Hub
    if push_to_hub and hub_model_id:
        print(f"📤 正在上传到HuggingFace Hub: {hub_model_id}")
        merged_model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print("✅ 上传完成!")
        return hub_model_id
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="合并LoRA adapter到基础模型")
    parser.add_argument("--adapter_model_id", type=str, 
                       default="yuexishen/codellama-7b-mbpp-ppo-qlora",
                       help="HuggingFace adapter模型ID")
    parser.add_argument("--output_dir", type=str, 
                       default="./merged_model",
                       help="输出目录")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="上传到HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str,
                       help="HuggingFace Hub模型ID")
    
    args = parser.parse_args()
    
    print("🚀 开始模型合并过程...")
    print(f"📂 Adapter模型: {args.adapter_model_id}")
    print(f"📁 输出目录: {args.output_dir}")
    
    if args.push_to_hub and not args.hub_model_id:
        print("❌ 错误: 要上传到Hub必须指定hub_model_id")
        return
    
    merged_model_path = merge_and_save_model(
        adapter_model_id=args.adapter_model_id,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id
    )
    
    print("🎯 合并完成! 现在可以用于评估:")
    print(f"Model path: {merged_model_path}")
    
    # 输出评估命令示例
    print("\n📋 BigCode评估命令:")
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

if __name__ == "__main__":
    main() 