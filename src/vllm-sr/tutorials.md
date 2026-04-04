# 构建项目
START_AFTER_BUILD=0 ./scripts/build_local_weak_network.sh

# 挂起服务
VLLM_SR_STACK_NAME=lane-b VLLM_SR_PORT_OFFSET=200 vllm-sr serve --minimal

# 停止
VLLM_SR_STACK_NAME=lane-b VLLM_SR_PORT_OFFSET=200 vllm-sr stop

# 意图分类测试

## 单次测试
curl -X POST http://localhost:8280/api/v1/classify/intent   -H "Content-Type: application/json"   -d '{"text": "what is cad?"}'

## 脚本
python ../../scripts/eval_classification_api_intent.py --router-url http://localhost:8280 --max-samples 1000

# 端到端测试
curl -v http://localhost:9099/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer sk-123456" -d '{ "model": "MoM", "messages": [{"role": "user", "content": "What is the derivative of x^2?"}] }'