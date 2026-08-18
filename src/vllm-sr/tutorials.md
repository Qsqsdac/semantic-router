# 构建项目
START_AFTER_BUILD=0 ./scripts/build_local_weak_network.sh

# 挂起服务
VLLM_SR_STACK_NAME=lane-b VLLM_SR_PORT_OFFSET=200 vllm-sr serve --minimal

# 停止
VLLM_SR_STACK_NAME=lane-b VLLM_SR_PORT_OFFSET=200 vllm-sr stop

# 意图分类测试

## 测试配置
使用 [config/unit_test/intent_test.yaml](../../config/unit_test/intent_test.yaml) 作为意图分类专用单元测试配置。

## 单次测试
curl -X POST http://localhost:8280/api/v1/classify/intent   -H "Content-Type: application/json"   -d '{"text": "what is cad?"}'

## 脚本
python ../../scripts/eval_intent_api.py --router-url http://localhost:8280 --max-samples 1000

# jailbreak 测试

## 测试配置
使用 [config/unit_test/jailbreak_test.yaml](../../config/unit_test/jailbreak_test.yaml) 作为 jailbreak 专用单元测试配置。

## 单次测试
curl -X POST http://localhost:8280/api/v1/classify/security   -H "Content-Type: application/json"   -d '{"text": "what is cad?"}'

## 脚本
python ../../scripts/eval_jailbreak_api.py --router-url http://localhost:8280 --max-samples 1000 --workers 4

# 事实核查测试

## 测试配置
使用 [config/unit_test/fact_check_test.yaml](../../config/unit_test/fact_check_test.yaml) 作为事实核查专用单元测试配置。

## 单次测试
curl -X POST http://localhost:8280/api/v1/classify/fact-check   -H "Content-Type: application/json"   -d '{"text": "Who is the first president of America?"}'

## 脚本
python ../../scripts/eval_fact_check_api.py --router-url http://localhost:8280 --max-samples 1000 --workers 4

# complexity 测试

## 测试配置
使用 [config/unit_test/complexity_test.yaml](../../config/unit_test/complexity_test.yaml) 作为 complexity 专用单元测试配置。

## 单次测试
curl -X POST http://localhost:8280/api/v1/classify/complexity   -H "Content-Type: application/json"   -d '{"text": "Explain the tradeoffs in consensus algorithms."}'

## 脚本
python ../../scripts/eval_complexity_api.py --router-url http://localhost:8280 --max-samples 1000 --workers 4

# 端到端测试

## Cerebras 中转模式

当容器内直连上游 API 受网络策略影响时，先在宿主机启动本地中转，再让路由调用本地中转：

```bash
# 启动中转服务（监听 0.0.0.0:18080）
python scripts/cerebras_openai_relay.py
```

中转健康检查：

```bash
curl -sS http://127.0.0.1:18080/healthz
curl -sS http://127.0.0.1:18080/v1/models | head
```

中转脚本：
- `scripts/cerebras_openai_relay.py`

路由配置默认已指向：
- `http://10.156.186.8:18080/v1`

## 单次测试
curl -v http://localhost:9099/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer sk-123456" -H "x-authz-user-id: demo-user" -H "x-authz-user-groups: premium-tier" -d '{ "model": "MoM", "messages": [{"role": "user", "content": "What is the derivative of x^2?"}]}'

## 脚本

### 分片测试
```bash
# 运行第 0 片（每片 500 条）
python ../../test/routerarena/routerarena_e2e_benchmark_chunked.py --slice-index 0 --output-subdir baseline
```

输出目录说明：
- 基础目录为 `reports/routerarena-e2e`
- 结果会写入 `reports/routerarena-e2e/<output-subdir>/`

### 非分片测试

```bash
python ../../test/routerarena/routerarena_e2e_benchmark.py --max-samples=100
```