#!/usr/bin/env python3
"""
下载脚本：一键下载所需模型
"""

import os
import urllib.request
import sys
from pathlib import Path

MODELS_DIR = Path("models")

def download_file(url: str, dest: Path, desc: str = ""):
    """带进度显示的文件下载"""
    print(f"下载 {desc}...")
    print(f"URL: {url}")
    print(f"保存至: {dest}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r进度: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest, progress_hook)
    print("\n完成！\n")

def download_llm_model():
    """下载 LLM 模型（使用 Hugging Face）"""
    print("=" * 50)
    print("1. 下载 LLM 模型")
    print("=" * 50)
    print("\n注意：Llama-3-8B 模型需要从 Hugging Face 下载")
    print("运行以下命令：")
    print("\n  huggingface-cli download unsloth/llama-3-8b-Instruct")
    print("\n或者使用 git-lfs：")
    print("  git clone https://huggingface.co/unsloth/llama-3-8b-Instruct models/llm")
    print("\n如果无法访问 Hugging Face，可以使用镜像：")
    print("  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...")

def download_whisper_model():
    """下载 Whisper.cpp 模型"""
    print("=" * 50)
    print("2. 下载 Whisper 语音模型")
    print("=" * 50)
    
    whisper_dir = MODELS_DIR / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    
    # Whisper 模型下载地址
    base_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
    
    models = {
        "tiny": "ggml-tiny.bin",      # 75 MB
        "base": "ggml-base.bin",      # 142 MB
        "small": "ggml-small.bin",    # 466 MB
        "medium": "ggml-medium.bin",  # 1.5 GB
    }
    
    print("\n可用模型大小：")
    for name, size in [("tiny", "75MB"), ("base", "142MB"), ("small", "466MB"), ("medium", "1.5GB")]:
        print(f"  - {name}: {size}")
    
    choice = input("\n选择要下载的模型 (tiny/base/small/medium，默认 small): ").strip() or "small"
    
    if choice not in models:
        print(f"未知选项: {choice}，使用默认 small")
        choice = "small"
    
    model_file = models[choice]
    url = f"{base_url}/{model_file}"
    dest = whisper_dir / model_file
    
    if dest.exists():
        print(f"\n模型已存在: {dest}")
        return
    
    try:
        download_file(url, dest, f"Whisper {choice} 模型")
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载：")
        print(f"  wget {url} -O {dest}")

def download_medical_ner():
    """下载医疗实体识别模型"""
    print("=" * 50)
    print("3. 下载医疗 NER 模型（可选）")
    print("=" * 50)
    print("\n医疗实体识别模型用于提取医学术语")
    print("可以从以下地址下载：")
    print("  - 中文医疗 BERT: https://huggingface.co/ckiplab/bert-base-chinese-ner")
    print("  - 或自建模型进行微调")

def main():
    """主函数"""
    print("""
╔═══════════════════════════════════════════╗
║   ClinicoScribe 模型下载工具              ║
╚═══════════════════════════════════════════╝

本脚本将帮助您下载运行所需的模型文件。
注意：模型文件较大，请确保有足够的磁盘空间。
""")
    
    # 创建目录
    MODELS_DIR.mkdir(exist_ok=True)
    
    # 下载 LLM 模型说明
    download_llm_model()
    
    # 下载 Whisper 模型
    try:
        download_whisper_model()
    except KeyboardInterrupt:
        print("\n\n用户取消下载")
    
    # 医疗 NER 说明
    download_medical_ner()
    
    print("\n" + "=" * 50)
    print("下载说明完成")
    print("=" * 50)
    print("\n下一步：")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 启动服务: python src/main.py")
    print("3. 测试 API: curl http://localhost:8000/health")

if __name__ == "__main__":
    main()
