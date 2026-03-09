"""
ClinicoScribe - 医疗文书小模型核心引擎
基于 Llama-3-8B 4-bit 量化版本，本地运行
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Optional
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalLLMEngine:
    """
    医疗专用小语言模型引擎
    使用量化技术降低显存占用，支持本地部署
    """
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_config(self, path: str) -> Dict:
        """加载模型配置"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_model(self):
        """加载 4-bit 量化模型"""
        model_name = self.config.get('model_name', 'unsloth/llama-3-8b-Instruct')
        
        logger.info(f"正在加载模型: {model_name}")
        
        # 4-bit 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        logger.info("模型加载完成")
    
    def generate_medical_record(
        self, 
        transcript: str, 
        department: str = "general",
        language: str = "zh"
    ) -> Dict[str, str]:
        """
        根据医生口述生成结构化病历
        
        Args:
            transcript: 医生口述的文本
            department: 科室（cardiology, orthopedics, emergency 等）
            language: 语言（zh/en）
        
        Returns:
            结构化病历字典
        """
        
        # 加载科室特定的 prompt 模板
        system_prompt = self._get_system_prompt(department, language)
        
        # 构建对话格式
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请将以下医生口述转换为结构化病历：\n\n{transcript}"}
        ]
        
        # 应用聊天模板
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 生成
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,  # 医疗文本需要确定性，温度调低
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # 解析为结构化数据
        return self._parse_medical_record(response)
    
    def _get_system_prompt(self, department: str, language: str) -> str:
        """获取科室特定的系统提示词"""
        
        base_prompt_zh = """你是一位经验丰富的医疗文书助手，专门负责将医生的口述转换为规范的病历记录。

要求：
1. 保持医学术语的准确性
2. 使用标准的 SOAP 格式或中国病历规范
3. 主诉要简明扼要（20字以内）
4. 现病史按时间顺序描述
5. 既往史分点列出
6. 不要添加诊断建议，仅整理已有信息
7. 如果信息不全，标注"待补充"

输出格式必须是以下JSON结构：
{
  "chief_complaint": "主诉",
  "present_illness": "现病史",
  "past_history": "既往史",
  "physical_exam": "体格检查",
  "auxiliary_exam": "辅助检查",
  "preliminary_diagnosis": "初步诊断",
  "treatment_plan": "处理意见"
}"""

        # 不同科室的特殊要求
        dept_prompts = {
            "cardiology": "重点关注心电图、心肌酶、血压、心率等心血管相关指标。",
            "orthopedics": "重点关注受伤机制、畸形、活动受限、影像学表现。",
            "emergency": "重点关注生命体征、意识状态、创伤评分、抢救措施。",
            "general": "按常规内科或外科病历格式整理。"
        }
        
        dept_note = dept_prompts.get(department, dept_prompts["general"])
        
        return base_prompt_zh + "\n\n" + dept_note
    
    def _parse_medical_record(self, text: str) -> Dict[str, str]:
        """解析模型输出为结构化数据"""
        import json
        import re
        
        # 尝试提取 JSON
        try:
            # 查找 JSON 代码块
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # 查找普通 JSON
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group(0))
                
        except json.JSONDecodeError:
            pass
        
        # 如果 JSON 解析失败，返回原始文本
        return {
            "raw_output": text,
            "chief_complaint": "",
            "present_illness": "",
            "note": "模型输出格式异常，请检查"
        }


if __name__ == "__main__":
    # 简单测试
    engine = MedicalLLMEngine()
    
    test_input = """
    患者女性 45 岁，主诉头晕 1 周。
    1 周来反复出现头晕，呈持续性，伴视物旋转，恶心，无呕吐。
    活动后加重，休息可缓解。
    既往体健，否认高血压、糖尿病史。
    查体：BP 120/80 mmHg，神清，颅神经检查未见异常。
    """
    
    result = engine.generate_medical_record(test_input, department="general")
    print(json.dumps(result, ensure_ascii=False, indent=2))
