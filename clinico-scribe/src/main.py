"""
FastAPI 服务主入口
提供 RESTful API 接口供前端或第三方系统调用
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import logging
import os
import tempfile
import shutil

from src.models.llm_engine import MedicalLLMEngine
from src.models.whisper_local import LocalWhisperASR, MockASR
from src.processors.soap_formatter import SOAPFormatter, TemplateManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="ClinicoScribe API",
    description="医疗文书 AI 助手 - 本地部署版",
    version="0.1.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例（延迟加载）
llm_engine: Optional[MedicalLLMEngine] = None
asr_engine = None
formatter = SOAPFormatter()
template_manager = TemplateManager()


def get_llm_engine():
    """获取或初始化 LLM 引擎"""
    global llm_engine
    if llm_engine is None:
        logger.info("初始化 LLM 引擎...")
        llm_engine = MedicalLLMEngine()
    return llm_engine


def get_asr_engine():
    """获取或初始化 ASR 引擎"""
    global asr_engine
    if asr_engine is None:
        # 检查是否有 whisper 模型，没有则使用 Mock
        whisper_model_path = "models/whisper/ggml-medium.bin"
        if os.path.exists(whisper_model_path):
            asr_engine = LocalWhisperASR()
        else:
            logger.warning("Whisper 模型未找到，使用 Mock ASR")
            asr_engine = MockASR()
    return asr_engine


# 数据模型
class TextToRecordRequest(BaseModel):
    text: str
    department: str = "general"
    language: str = "zh"
    output_format: str = "json"  # json, soap, hl7


class RecordResponse(BaseModel):
    success: bool
    data: Optional[Dict]
    message: str
    missing_fields: Optional[list] = None


# 健康检查
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ClinicoScribe",
        "version": "0.1.0"
    }


# 文本转病历
@app.post("/generate", response_model=RecordResponse)
async def text_to_record(request: TextToRecordRequest):
    """
    将医生口述文本转换为结构化病历
    """
    try:
        engine = get_llm_engine()
        
        # 生成结构化病历
        record = engine.generate_medical_record(
            transcript=request.text,
            department=request.department,
            language=request.language
        )
        
        # 验证完整性
        missing = template_manager.validate_record(record, request.department)
        
        # 根据要求格式化输出
        if request.output_format == "soap":
            formatted = formatter.format(record, request.department)
            record["formatted_text"] = formatted
        elif request.output_format == "hl7":
            formatted = formatter.format_hl7(record)
            record["hl7_message"] = formatted
        
        return RecordResponse(
            success=True,
            data=record,
            message="病历生成成功",
            missing_fields=missing if missing else None
        )
        
    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        return RecordResponse(
            success=False,
            data=None,
            message=f"生成失败: {str(e)}"
        )


# 语音转病历
@app.post("/transcribe")
async def audio_to_record(
    audio: UploadFile = File(...),
    department: str = Form("general"),
    language: str = Form("zh")
):
    """
    上传音频文件，自动转录并生成病历
    支持 wav, mp3, m4a 格式
    """
    temp_path = None
    
    try:
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f".{audio.filename.split('.')[-1]}"
        ) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            temp_path = tmp.name
        
        # 语音识别
        asr = get_asr_engine()
        transcript = asr.transcribe(temp_path)
        
        if not transcript:
            raise HTTPException(status_code=400, detail="语音识别失败")
        
        # 生成病历
        engine = get_llm_engine()
        record = engine.generate_medical_record(
            transcript=transcript,
            department=department,
            language=language
        )
        
        # 验证完整性
        missing = template_manager.validate_record(record, department)
        
        return {
            "success": True,
            "transcript": transcript,  # 返回转录文本供医生核对
            "record": record,
            "missing_fields": missing if missing else None
        }
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# 获取支持的科室列表
@app.get("/departments")
async def list_departments():
    """获取支持的科室列表和配置"""
    return {
        "departments": [
            {
                "id": "general",
                "name": "通用内科/外科",
                "description": "适用于一般门诊"
            },
            {
                "id": "cardiology",
                "name": "心内科",
                "description": "包含心电图、心肌酶等心血管专项"
            },
            {
                "id": "orthopedics",
                "name": "骨科",
                "description": "包含受伤机制、影像学专项"
            },
            {
                "id": "emergency",
                "name": "急诊科",
                "description": "包含生命体征、抢救措施专项"
            }
        ]
    }


# 获取病历模板
@app.get("/template/{department}")
async def get_template(department: str):
    """获取指定科室的病历模板结构"""
    template = template_manager.get_template(department)
    return {
        "department": department,
        "required_fields": template["required_fields"],
        "focus_areas": template["focus_areas"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
