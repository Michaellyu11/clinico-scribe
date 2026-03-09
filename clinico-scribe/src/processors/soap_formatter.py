"""
病历格式化器
将模型输出转换为标准病历格式
"""

from typing import Dict, List
import json
from datetime import datetime


class SOAPFormatter:
    """
    SOAP 病历格式化器
    支持中国病历规范和电子病历标准
    """
    
    def __init__(self):
        self.template = """
【病历记录】

就诊时间：{visit_time}
科室：{department}

一、主诉（Chief Complaint）
{chief_complaint}

二、现病史（History of Present Illness）
{present_illness}

三、既往史（Past History）
{past_history}

四、体格检查（Physical Examination）
{physical_exam}

五、辅助检查（Auxiliary Examinations）
{auxiliary_exam}

六、初步诊断（Preliminary Diagnosis）
{preliminary_diagnosis}

七、处理意见（Treatment Plan）
{treatment_plan}

医师签名：__________
记录时间：{record_time}
"""
    
    def format(self, structured_data: Dict, department: str = "内科") -> str:
        """
        将结构化数据格式化为完整病历
        """
        now = datetime.now()
        
        context = {
            "visit_time": now.strftime("%Y-%m-%d %H:%M"),
            "department": department,
            "chief_complaint": structured_data.get("chief_complaint", ""),
            "present_illness": structured_data.get("present_illness", ""),
            "past_history": structured_data.get("past_history", ""),
            "physical_exam": structured_data.get("physical_exam", ""),
            "auxiliary_exam": structured_data.get("auxiliary_exam", ""),
            "preliminary_diagnosis": structured_data.get("preliminary_diagnosis", ""),
            "treatment_plan": structured_data.get("treatment_plan", ""),
            "record_time": now.strftime("%Y-%m-%d %H:%M")
        }
        
        return self.template.format(**context)
    
    def format_json(self, structured_data: Dict) -> str:
        """输出 JSON 格式（用于系统对接）"""
        return json.dumps(structured_data, ensure_ascii=False, indent=2)
    
    def format_hl7(self, structured_data: Dict) -> str:
        """
        输出 HL7 FHIR 格式（用于医院信息系统对接）
        简化版实现
        """
        patient_id = "P" + datetime.now().strftime("%Y%m%d%H%M%S")
        
        hl7_template = f"""MSH|^~\\&|CLINICO_SCRIBE|HOSPITAL|HIS|HOSPITAL|{datetime.now().strftime('%Y%m%d%H%M%S')}||MDM^T02|{patient_id}|P|2.5
PID|1||{patient_id}||^||||||||||||||||||
PV1|1|O|||||||||||||||||||||||||||||||||
TXA|1|CN|TX|||{datetime.now().strftime('%Y%m%d%H%M%S')}|||||||||||{structured_data.get('chief_complaint', '')}
OBX|1|TX|||{structured_data.get('present_illness', '')}||||||F
"""
        return hl7_template


class TemplateManager:
    """
    科室病历模板管理器
    不同科室有不同的关注点和格式要求
    """
    
    def __init__(self):
        self.templates = {
            "cardiology": {
                "required_fields": [
                    "chief_complaint", "present_illness", "past_history",
                    "physical_exam", "ecg_findings", "cardiac_enzymes",
                    "preliminary_diagnosis", "treatment_plan"
                ],
                "focus_areas": ["胸痛特点", "心电图", "心肌酶", "血压"]
            },
            "orthopedics": {
                "required_fields": [
                    "chief_complaint", "injury_mechanism", "physical_exam",
                    "imaging_findings", "preliminary_diagnosis", "treatment_plan"
                ],
                "focus_areas": ["受伤机制", "畸形", "活动度", "影像学"]
            },
            "emergency": {
                "required_fields": [
                    "chief_complaint", "present_illness", "vital_signs",
                    "consciousness", "emergency_treatment", "preliminary_diagnosis"
                ],
                "focus_areas": ["生命体征", "意识状态", "创伤评分", "抢救措施"]
            },
            "general": {
                "required_fields": [
                    "chief_complaint", "present_illness", "past_history",
                    "physical_exam", "auxiliary_exam", "preliminary_diagnosis", "treatment_plan"
                ],
                "focus_areas": ["一般情况", "系统回顾"]
            }
        }
    
    def get_template(self, department: str) -> Dict:
        """获取科室模板"""
        return self.templates.get(department, self.templates["general"])
    
    def validate_record(self, record: Dict, department: str) -> List[str]:
        """验证病历是否完整"""
        template = self.get_template(department)
        missing = []
        
        for field in template["required_fields"]:
            if field not in record or not record[field]:
                missing.append(field)
        
        return missing
