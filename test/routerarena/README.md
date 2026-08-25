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
