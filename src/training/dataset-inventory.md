# 数据集清单（简版）

本清单只保留项目里常用、且代码中有明确来源的数据集。

## 1. 路由信号评测（dashboard/signal_eval）

| 数据集ID | 用途 | 来源 | 获取方式 |
|---|---|---|---|
| mmlu-pro-en | 领域分类评测（英文） | TIGER-Lab/MMLU-Pro | `load_dataset("TIGER-Lab/MMLU-Pro", split="test")` |
| mmlu-prox-<lang> | 领域分类评测（多语言） | li-lab/MMLU-ProX | `load_dataset("li-lab/MMLU-ProX", "zh", split="test")`（语言代码替换） |
| fact-check-en | 事实核查信号评测 | llm-semantic-router/fact-check-classification-dataset | `load_dataset("llm-semantic-router/fact-check-classification-dataset", split="test")` |
| feedback-en | 用户反馈信号评测 | llm-semantic-router/feedback-detector-dataset | `load_dataset("llm-semantic-router/feedback-detector-dataset", split="validation")` |

参考实现：
- `src/training/model_eval/signal_eval.py`
- `dashboard/backend/evaluation/runner.go`

## 2. 训练数据（分类模型）

### Intent 分类

- `TIGER-Lab/MMLU-Pro`
- `LLM-Semantic-Router/category-classifier-supplement`

获取：

```python
from datasets import load_dataset
load_dataset("TIGER-Lab/MMLU-Pro")
load_dataset("LLM-Semantic-Router/category-classifier-supplement")
```

参考实现：
- `src/training/model_classifier/classifier_model_fine_tuning_lora/ft_linear_lora.py`

### PII 检测

- `ai4privacy/pii-masking-400k`（HF）
- Presidio 合成数据（GitHub 原始 JSON）

Presidio 链接（代码里使用）：
- `https://raw.githubusercontent.com/microsoft/presidio-research/refs/heads/master/data/synth_dataset_v2.json`

获取：

```bash
python - <<'PY'
from datasets import load_dataset
load_dataset("ai4privacy/pii-masking-400k", split="train")
print("AI4Privacy OK")
PY

curl -L -o presidio_synth_dataset_v2.json \
  https://raw.githubusercontent.com/microsoft/presidio-research/refs/heads/master/data/synth_dataset_v2.json
```

参考实现：
- `src/training/model_classifier/pii_model_fine_tuning_lora/pii_bert_finetuning_lora.py`

## 3. 幻觉评测（bench/hallucination）

- `halueval` -> `pminervini/HaluEval`（`qa_samples` / `data`）
- `financebench` -> `PatronusAI/financebench`
- 也支持本地 JSONL

获取：

```python
from datasets import load_dataset
load_dataset("pminervini/HaluEval", "qa_samples", split="data")
load_dataset("PatronusAI/financebench")
```

参考实现：
- `bench/hallucination/datasets.py`

## 4. 推理基准（bench/reasoning）

常见来源：
- `TIGER-Lab/MMLU-Pro`
- `Idavidrein/gpqa`
- `allenai/ai2_arc`
- `gsm8k`
- `truthful_qa`
- `openbookqa`

参考实现：
- `bench/reasoning/dataset_implementations/*.py`

## 5. 模型选择训练数据

- 数据集：`vllm-project/semantic-router-benchmark`
- 文件：`benchmark_training_data.jsonl`

获取：

```bash
huggingface-cli download vllm-project/semantic-router-benchmark \
  --repo-type dataset \
  --include benchmark_training_data.jsonl \
  --local-dir .cache/ml_model_selection
```

参考实现：
- `src/training/model_selection/ml_model_selection/data_loader.py`

## 6. 这些数据集如何参与训练（模型对应）

| 数据集 | 对应模型/任务 | 参与阶段 | 说明 |
|---|---|---|---|
| TIGER-Lab/MMLU-Pro | mmbert32k-intent-classifier（领域分类） | 训练 + 评测 | Intent 主训练集；也用于 domain 信号评测。 |
| LLM-Semantic-Router/category-classifier-supplement | mmbert32k-intent-classifier | 训练 | 作为补充样本，增强 other 类识别。 |
| ai4privacy/pii-masking-400k | mmbert32k-pii-detector | 训练 | PII 主体数据之一，通常与 Presidio 混合。 |
| Presidio synth_dataset_v2.json | mmbert32k-pii-detector | 训练 | 提供高质量 span 标注，与 AI4Privacy 组合训练。 |
| li-lab/MMLU-ProX | 路由 domain 信号（多语言） | 评测 | 当前主要用于多语言评测，不是主训练集。 |
| llm-semantic-router/fact-check-classification-dataset | 路由 fact_check 信号 | 评测 | 用于 fact-check 信号评测与回归检查。 |
| llm-semantic-router/feedback-detector-dataset | mmbert32k-feedback-detector | 训练 + 评测 | 反馈分类器训练默认数据源，也用于信号评测。 |
| pminervini/HaluEval | 幻觉检测基准；fact-check 数据构造 | 评测 + 训练辅助 | 在 hallucination benchmark 中直接评测；也被 fact-check 训练脚本用作问题来源之一。 |
| PatronusAI/financebench | 幻觉检测基准 | 评测 | 用于 hallucination benchmark 的金融场景评测。 |
| vllm-project/semantic-router-benchmark | ML model selection 模型 | 训练 | 作为训练样本，学习 query 到最优模型选择策略。 |

补充：
- fact-check 分类器训练脚本还会混合 SQuAD、TriviaQA、HotpotQA、TruthfulQA、CoQA、FactCHD 等数据来构造训练样本。