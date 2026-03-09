# ClinicoScribe

> 为临床医生打造的本地化 AI 文书助手

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 🩺 为什么做这个？

医生 40% 的时间花在写病历上。大模型虽然强，但：
- 需要联网（医院内网不行）
- 贵（GPT-4 API 成本高）
- 有隐私风险（患者数据不能外传）

**ClinicoScribe** 是一个可以本地运行的小语言模型（SLM），专门做一件事：**把医生的口述转成结构化病历**。

## ✨ 核心功能

- 🎙️ **语音转病历** - 医生口述，自动生成 SOAP 格式病历
- 📝 **智能结构化** - 自动提取主诉、现病史、用药史
- 🔒 **完全本地** - 数据不出医院，隐私合规
- 🏥 **专科定制** - 心内科、骨科、急诊科等专科模板
- 💰 **低成本** - 单张 4090 即可运行，无需联网

## 🚀 快速开始

### 环境要求
- Python 3.9+
- CUDA 11.8+ (推荐显存 8GB+)
- Linux/macOS/Windows

### 安装

```bash
git clone https://github.com/yourusername/clinico-scribe.git
cd clinico-scribe
pip install -r requirements.txt

# 下载模型（约 4GB）
python scripts/download_model.py
```

### 运行演示

```bash
# 启动 API 服务
python src/main.py

# 测试语音转病历
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "examples/doctor_voice_sample.wav",
    "department": "cardiology",
    "language": "zh"
  }'
```

## 📁 项目结构

```
clinico-scribe/
├── src/
│   ├── main.py              # FastAPI 服务入口
│   ├── models/              # 模型加载与推理
│   │   ├── llm_engine.py    # SLM 核心推理
│   │   ├── whisper_local.py # 本地语音识别
│   │   └── medical_ner.py   # 医疗实体识别
│   ├── processors/          # 病历处理逻辑
│   │   ├── soap_formatter.py
│   │   └── template_manager.py
│   └── utils/
├── models/                  # 存放下载的模型文件
├── data/
│   ├── templates/           # 专科病历模板
│   └── examples/            # 示例音频和病历
├── configs/
│   ├── model_config.yaml    # 模型配置
│   └── departments/         # 各科室配置
└── tests/                   # 单元测试
```

## 🏗️ 技术架构

```
医生语音输入
    ↓
Whisper.cpp (本地 ASR)
    ↓
文本预处理 → 医疗实体识别 (BERT-based)
    ↓
Llama-3-8B-Instruct (4-bit 量化)
    ↓
结构化病历输出 (JSON/SOAP)
```

## 💡 使用示例

### 示例 1：门诊病历生成

```python
from clinico_scribe import MedicalScribe

scribe = MedicalScribe(department="cardiology")

# 模拟医生口述
doctor_speech = """
患者男性 58 岁，主诉胸痛 3 小时。
患者今日上午 10 点突发胸骨后压榨性疼痛，
向左肩放射，伴大汗、恶心。
既往高血压 10 年，糖尿病 5 年，吸烟史 30 年。
查体：BP 160/95 mmHg，HR 92 次/分，心肺未见明显异常。
心电图示 V1-V4 ST 段抬高。
"""

record = scribe.generate_record(doctor_speech)
print(record)
```

**输出：**
```json
{
  "chief_complaint": "胸痛 3 小时",
  "present_illness": "患者男性，58岁。今上午10时突发胸骨后压榨性疼痛，向左肩放射，伴大汗、恶心...",
  "past_history": "高血压病史10年，糖尿病病史5年，吸烟史30年（20支/日）",
  "physical_exam": "BP 160/95 mmHg，P 92次/分。心肺查体未见明显异常。",
  "auxiliary_exam": "心电图：V1-V4导联ST段弓背向上抬高",
  "preliminary_diagnosis": "1. 急性广泛前壁心肌梗死；2. 高血压病3级（极高危）；3. 2型糖尿病",
  "treatment_plan": "1. 立即给予阿司匹林300mg嚼服；2. 建立静脉通道；3. 急查心肌酶谱、凝血功能；4. 请心内科会诊，评估急诊PCI指征"
}
```

## 🎯 Roadmap

- [x] 基础病历结构化生成
- [x] 本地 Whisper 语音识别
- [ ] 多专科模板（心内、骨科、急诊）
- [ ] 与医院 HIS 系统对接
- [ ] 语音实时转录（流式处理）
- [ ] 医学术语自动补全

## ⚠️ 免责声明

**本项目仅用于辅助病历文书工作，不构成医疗诊断建议。**

- 生成的病历需经医生审核确认
- 不用于诊断决策，仅用于文档整理
- 使用需符合当地医疗法规（如 HIPAA）

## 🤝 贡献

欢迎提交 Issue 和 PR！特别是：
- 更多专科的病历模板
- 医疗实体识别训练数据
- 医院 HIS 系统对接经验

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

---

**⭐ 如果这个项目对你有帮助，请点个 Star！**