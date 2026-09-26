# semantic-router 的 RouterArena 端到端评测

这个目录包含一个基于 RouterArena 的端到端评测脚本，调用的是 src/vllm-sr/tutorials.md 中描述的 OpenAI 兼容聊天接口。

脚本会从 Hugging Face 加载 RouterArena 的 parquet split，按 RouterArena zero-shot 配置构造题目 prompt，只通过 POST /v1/chat/completions 发起请求，并将逐条结果与汇总结果写入 reports/routerarena-e2e/。

默认会同时评测 full 和 robustness 两个 split。

## 运行方式

脚本采用串行单请求模式，逐条发送请求以测量真实延迟（无并发）。

```bash
python test/routerarena/routerarena_e2e_benchmark.py \
  --router-url http://localhost:9099 \
  --splits full robustness
```

## 指定样本量运行

使用参数 --max-samples 可以限制每个 split 的样本数。

- --max-samples 0：不限制样本量（默认值）
- --max-samples N：每个 split 最多跑 N 条

示例 1：full 和 robustness 各跑 100 条

```bash
python test/routerarena/routerarena_e2e_benchmark.py \
  --router-url http://localhost:9099 \
  --splits full robustness \
  --max-samples 100
```

示例 2：只跑 full，并限制为 200 条

```bash
python test/routerarena/routerarena_e2e_benchmark.py \
  --router-url http://localhost:9099 \
  --splits full \
  --max-samples 200
```

## 输出指标

- 逐条样本输出：模型回答、路由到的模型、任务分数
- 路由耗时：读取响应头 x-vsr-total-routing-latency-ms
- 汇总耗时：平均值、P50、P95、P99
- 鲁棒性：对比 full 与 robustness 重叠样本的模型选择翻转率/稳定性

说明：LiveCodeBench 数据会纳入评测流程，但如果缺少代码执行依赖，其精确代码执行分数会标记为 unsupported。

## RouteLLM baseline

RouteLLM baseline 使用完全相同的 RouterArena 数据集、split 排布、zero-shot prompt 和评分函数。当前配置包含 `mf`、`bert` 和 `causal_llm` 三种路由器。路由器在进程内执行，因此 detail 文件会额外记录 `route_decision`、`router_score`、`threshold` 和 `routing_latency_ms`；最终回答仍记录 `selected_model`、`task_score` 和 `status`。

默认实际后端模型与 `src/vllm-sr/config.yaml` 一致：strong 为 `Qwen/Qwen3.5-27B`，weak 为 `Qwen/Qwen3.5-4B`。RouteLLM MF checkpoint 的训练模型 ID 通过 `--route-strong-model`/`--route-weak-model` 单独配置，避免与实际推理模型混淆。

```bash
cd /home/chengsixiang/semantic-router
PYTHONPATH=/home/chengsixiang/RouteLLM:$PYTHONPATH \
  python test/routerarena/routellm_e2e_benchmark.py \
  --splits full robustness \
  --backend-url http://10.156.186.8:18081/v1 \
  --embedding-base-url https://api.openai.com/v1 \
  --embedding-api-key "$OPENAI_API_KEY" \
  --max-samples 100
```

输出写入 `reports/routerarena-e2e/routellm/`，包含每个 split 的 JSONL detail、`latest_*_detail.jsonl`、带时间戳 summary 和 `latest_summary.json`。

切换路由器时使用 `--router bert` 或 `--router causal_llm`；对应 checkpoint 已在 `routellm_baseline.yaml` 中配置。

## M-IRT baseline

M-IRT baseline 运行在独立的 OpenAI-compatible gateway 中：它固定使用
`mirt_bert.snapshot`、`bert-base-uncased` 的固定 commit、同一 BERT 的
mean pooling 与 L2 normalization，并从 Git LFS 取回的 `query.csv` 和
`query_embeddings.pkl` 的训练部分建立 KNN 冷启动索引。新请求使用 5 个
近邻 query 的平均向量按固定 `lambda` 融合；候选模型 profile 也使用同一
BERT 编码并缓存。新候选模型不需要表现标签或重新训练 M-IRT，但这是需要
单独报告的实验性 zero-shot 泛化假设。

启动 gateway：

```bash
cd /home/chengsixiang/IRT-Router
python mirt_service.py --config router_config.yaml
```

运行与 RouteLLM 相同的 RouterArena 数据、prompt、split 对齐和评分协议：

```bash
cd /home/chengsixiang/semantic-router
python test/routerarena/mirt_e2e_benchmark.py \
  --router-url http://127.0.0.1:8088 \
  --splits full robustness \
  --max-samples 100
```

`a`、`lambda`、价格快照、候选集合和 profile 文本均在 gateway 的
`router_config.yaml` 中固定，不作为客户端参数。gateway 记录 request ID、
候选集版本、预测概率、utility、估算成本、实际成本、延迟和失败原因；
这些记录不反哺训练。
