"""
本地语音识别模块
基于 Whisper.cpp，完全本地运行，无需联网
"""

import os
import subprocess
import tempfile
import logging
from typing import Optional
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


class LocalWhisperASR:
    """
    本地语音识别引擎
    使用 Whisper.cpp 实现，支持中文和英文医学术语识别
    """
    
    def __init__(self, model_size: str = "medium", language: str = "zh"):
        """
        Args:
            model_size: tiny/base/small/medium/large
            language: 语言代码 (zh/en)
        """
        self.model_size = model_size
        self.language = language
        self.model_path = f"models/whisper/ggml-{model_size}.bin"
        self._check_model()
    
    def _check_model(self):
        """检查模型文件是否存在"""
        if not os.path.exists(self.model_path):
            logger.warning(f"Whisper 模型未找到: {self.model_path}")
            logger.info("请运行: python scripts/download_whisper_model.py")
    
    def transcribe(
        self, 
        audio_path: str, 
        medical_mode: bool = True
    ) -> str:
        """
        转录音频为文本
        
        Args:
            audio_path: 音频文件路径 (wav/mp3)
            medical_mode: 是否启用医学术语增强
        
        Returns:
            转录文本
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 转换为 16kHz 单声道 wav（Whisper 要求）
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            processed_path = tmp.name
        
        try:
            self._preprocess_audio(audio_path, processed_path)
            
            # 构建 whisper.cpp 命令
            cmd = [
                "./whisper.cpp/main",  # 假设 whisper.cpp 已编译
                "-m", self.model_path,
                "-f", processed_path,
                "-l", self.language,
                "--output-txt",
                "-pp"  # 打印进度
            ]
            
            # 如果是医学模式，添加热词（未来可以扩展）
            if medical_mode:
                # 可以在这里添加医学术语热词
                pass
            
            logger.info(f"开始转录: {audio_path}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode != 0:
                logger.error(f"转录失败: {result.stderr}")
                return ""
            
            # 读取结果
            transcript = self._extract_text(result.stdout)
            logger.info(f"转录完成，文本长度: {len(transcript)}")
            return transcript
            
        finally:
            # 清理临时文件
            if os.path.exists(processed_path):
                os.unlink(processed_path)
    
    def _preprocess_audio(self, input_path: str, output_path: str):
        """音频预处理：转为 16kHz 单声道"""
        import librosa
        
        # 加载音频
        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        
        # 保存
        sf.write(output_path, audio, 16000)
    
    def _extract_text(self, stdout: str) -> str:
        """从 whisper.cpp 输出中提取文本"""
        lines = stdout.strip().split('\n')
        
        # 过滤时间戳，只保留文本
        text_lines = []
        for line in lines:
            # 跳过时间戳行 [00:00:00.000 --> 00:00:05.000]
            if '-->' in line or line.startswith('['):
                continue
            if line.strip():
                text_lines.append(line.strip())
        
        return ' '.join(text_lines)
    
    def transcribe_realtime(self, duration: int = 30) -> str:
        """
        实时转录（从麦克风）
        用于医生边说话边生成病历的场景
        
        Args:
            duration: 录音时长（秒）
        """
        import sounddevice as sd
        
        logger.info(f"开始录音，时长 {duration} 秒...")
        
        # 录音
        sample_rate = 16000
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            temp_path = tmp.name
        
        try:
            return self.transcribe(temp_path)
        finally:
            os.unlink(temp_path)


class MockASR:
    """
    模拟 ASR（用于测试，无需下载模型）
    """
    
    def transcribe(self, audio_path: str, **kwargs) -> str:
        """返回模拟文本"""
        return """
        患者男性 58 岁，主诉胸痛 3 小时。
        今日上午突发胸骨后压榨性疼痛，向左肩放射。
        既往高血压 10 年，吸烟史 30 年。
        查体血压 160 95，心率 92 次每分。
        心电图示 ST 段抬高。
        """
