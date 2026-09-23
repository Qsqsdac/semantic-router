# OpenAI 记录与回放后端

`openai_record_replay_backend.py` 是用于可重复路由实验的 OpenAI 兼容 HTTP 后端。它接收聊天补全请求；当完整请求与已存储记录匹配时回放响应；缓存未命中时可将请求转发给真实的 OpenAI 兼容上游。

缓存键是规范化请求 JSON 的 SHA-256 哈希，包含路由器选定的物理 `model`、消息及其他所有影响生成结果的请求字段。因此，路由决策变更会有意导致缓存未命中。若物理模型 ID 与上游供应商的 ID 不同，服务仅在转发时改写模型名，缓存键仍使用原始物理模型 ID。

## 接口

| 路径 | 用途 |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI 兼容的聊天补全接口 |
| `POST /chat/completions` | 兼容语义路由服务重写后的路径 |
| `GET /healthz` | 存活检查，返回当前缓存模式 |
| `GET /v1/models` | 最小化的 OpenAI models 响应 |

响应包含 `X-OpenAI-Replay` 头，其值为 `HIT`、`MISS; id=<cache-key>` 或 `BYPASS; upstream_status=<status>`。仅成功的上游响应会写入缓存；非成功响应以 `BYPASS` 透传且不会被缓存。回放响应在保存了原始请求耗时时还会包含 `X-OpenAI-Replay-Original-Elapsed-Ms`。

## 当前运行的服务

当前服务运行在 tmux 会话 `openai-replay` 中，以 `auto` 模式监听 `0.0.0.0:18081`。实际 Python 进程为：

```bash
/home/chengsixiang/.venv/bin/python \
  src/vllm-sr/scripts/openai_record_replay_backend.py --mode auto
```

tmux 启动器会将输出追加至 `/tmp/openai-replay.log`。脚本默认值补全了未显式传入的设置：

| 设置 | 当前生效值 |
| --- | --- |
| 监听地址 | `0.0.0.0` |
| 端口 | `18081` |
| 模式 | `auto` |
| 缓存目录 | `.cache/openai-replay`，相对于启动目录 |
| 上游超时 | `600` 秒 |

查看、连接到正在运行的服务：

```bash
tmux ls
tmux attach -t openai-replay
tail -f /tmp/openai-replay.log
curl http://127.0.0.1:18081/healthz
```

使用 `Ctrl-b d` 从 tmux 分离。停止服务：

```bash
tmux kill-session -t openai-replay
```

通过下文的受控启动器启动时，销毁 `openai-replay` 会话会同时停止本机 Qwen3.5-4B 进程并释放显存。

## 启动命令

在仓库根目录执行以下命令。请求可能缓存未命中时（`auto` 或 `record` 模式），未匹配专用模型规则的请求必须设置默认上游 URL 和 API 密钥。此启动器会先启动本机 Qwen3.5-4B，确认健康后再启动 replay；replay 退出、被终止或收到挂起信号时会连带停止或挂起 4B 服务。

```bash
cd /home/chengsixiang/semantic-router

tmux new-session -d -s openai-replay \
  'REPLAY_PYTHON=/home/chengsixiang/.venv/bin/python \
   QWEN35_4B_PYTHON=/home/chengsixiang/RouteLLM/.venv/bin/python \
   QWEN35_4B_SERVICE_DIR=/home/chengsixiang/qwen35-4b-service \
   REPLAY_HOST=0.0.0.0 \
   REPLAY_PORT=18081 \
   REPLAY_CACHE_DIR=/home/chengsixiang/semantic-router/.cache/openai-replay \
   REPLAY_MODE=auto \
   REPLAY_UPSTREAM_BASE_URL=https://<upstream-host>/v1 \
   REPLAY_UPSTREAM_API_KEY=<api-key> \
   REPLAY_UPSTREAM_TIMEOUT_SECONDS=600 \
   src/vllm-sr/scripts/start_openai_replay_with_local_qwen4b.sh \
     --mode auto \
     2>&1 | tee -a /tmp/openai-replay.log'
```

不要在提交到仓库的脚本中写入凭据。交互式启动时，应在创建 tmux 会话前先在 Shell 中导出 `REPLAY_UPSTREAM_API_KEY`。

若 `DASHSCOPE_API_KEY` 仅定义在 `~/.bashrc`，请在创建 tmux 会话前将它导出到当前 Shell；`tmux new-session '...'` 使用的非交互 shell 不会读取 `~/.bashrc`，但会继承创建会话时已有的环境变量。不要为此盲目加载完整的 `~/.bashrc`，因为它可能包含其他交互式初始化操作；只导出所需变量后再执行启动命令。

缓存未命中时，Qwen3.5 模型会使用专用上游：

| 模型 | 默认上游 | 密钥来源 |
| --- | --- | --- |
| `Qwen/Qwen3.5-27B` 或 `qwen3.5-27b` | 阿里云百炼 OpenAI 兼容接口；出站模型 ID 固定为 `qwen3.5-27b` | `DASHSCOPE_API_KEY` |
| `Qwen/Qwen3.5-4B` 或 `qwen3.5-4b` | `http://127.0.0.1:18080/v1` | 无 |

两条规则都可通过环境变量覆盖：`REPLAY_QWEN35_27B_UPSTREAM_BASE_URL`、`REPLAY_QWEN35_27B_UPSTREAM_API_KEY` 和 `REPLAY_QWEN35_4B_UPSTREAM_BASE_URL`。其他模型仍使用 `REPLAY_UPSTREAM_BASE_URL` 与 `REPLAY_UPSTREAM_API_KEY`。

## 模式

| 模式 | 已有记录 | 缓存未命中 |
| --- | --- | --- |
| `replay` | 直接返回记录 | 返回 `404 replay_miss`，绝不调用上游 |
| `auto` | 直接返回记录 | 转发到上游、持久化响应，再返回响应 |
| `record` | 忽略已有记录 | 始终转发到上游，并替换原记录 |

使用 `replay` 执行确定性的无网络评估。使用 `auto` 复用全部兼容的历史生成结果，同时允许新的请求回源。仅在需要有意刷新保存结果时使用 `record`。

## 路由器集成

当前路由器配置将两个模型后端都指向此服务：

```yaml
endpoint: "10.156.186.8:18081"
base_url: "http://10.156.186.8:18081/v1"
chat_path: /chat/completions
```

除 `base_url` 外还必须设置 `endpoint`：这可使 Envoy 获得独立的主机和端口字段，而不会将 `host:port` 误认为主机名。服务同时接受配置中的 `/v1/chat/completions` 基础路径和 Envoy 重写后的 `/chat/completions` 路径。

语义路由服务自身的语义缓存可以在请求到达此服务前直接满足请求。因此，路由器请求量不一定等于 replay 服务请求量。

## 配置参考

CLI 参数：

| 选项 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--host` | `REPLAY_HOST` | `0.0.0.0` | 监听地址 |
| `--port` | `REPLAY_PORT` | `18081` | 监听端口 |
| `--cache-dir` | `REPLAY_CACHE_DIR` | `.cache/openai-replay` | 存放 JSON 记录的目录 |
| `--mode` | `REPLAY_MODE` | `auto` | `auto`、`record` 或 `replay` |
| `--import-routerarena-detail` | 无 | 无 | 从 RouterArena detail JSONL 文件导入成功记录 |
| `--reasoning-effort` | 无 | 空 | 导入历史记录时添加该请求字段 |

仅用于转发的环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `REPLAY_UPSTREAM_BASE_URL` | 空 | 上游 OpenAI 基础 URL，包含 `/v1` 后缀 |
| `REPLAY_UPSTREAM_API_KEY` | 空 | 以 `Authorization: Bearer <key>` 发送给上游 |
| `REPLAY_QWEN35_27B_UPSTREAM_BASE_URL` | 阿里云百炼兼容接口 | Qwen3.5-27B 专用 OpenAI 基础 URL |
| `REPLAY_QWEN35_27B_UPSTREAM_API_KEY` | `DASHSCOPE_API_KEY` | Qwen3.5-27B 专用 API 密钥 |
| `REPLAY_QWEN35_4B_UPSTREAM_BASE_URL` | `http://127.0.0.1:18080/v1` | Qwen3.5-4B 本机服务基础 URL |
| `REPLAY_PYTHON` | 无 | 运行 replay 服务的 Python 解释器；受控启动器必填 |
| `QWEN35_4B_PYTHON` | 无 | 运行本机 Qwen3.5-4B 服务的 Python 解释器；受控启动器必填 |
| `QWEN35_4B_SERVICE_DIR` | 无 | 本机 `qwen35-4b-service` 目录；受控启动器必填 |
| `QWEN35_4B_HOST` / `QWEN35_4B_PORT` | `127.0.0.1` / `18080` | 本机 Qwen3.5-4B 服务监听地址 |
| `QWEN35_4B_STARTUP_TIMEOUT_SECONDS` | `180` | 等待本机 Qwen3.5-4B 健康检查的秒数 |
| `REPLAY_UPSTREAM_TIMEOUT_SECONDS` | `600` | 上游 HTTP 超时时间，单位为秒 |
| `REPLAY_UPSTREAM_PROXY` | 空 | 可选代理 URL；默认会忽略代理环境变量。 |

## 导入历史 RouterArena 结果

仅导入成功的 detail 行：`http_status == 200`，且行中必须包含 prompt、选定的物理模型和原始响应。

```bash
cd /home/chengsixiang/semantic-router
/home/chengsixiang/.venv/bin/python \
  src/vllm-sr/scripts/openai_record_replay_backend.py \
  --cache-dir .cache/openai-replay \
  --import-routerarena-detail \
    reports/routerarena-e2e/routerarena_e2e_full_detail_20260903-110803.jsonl \
  --reasoning-effort <value-used-by-the-benchmark>
```

导入的请求必须与 benchmark 请求精确匹配。尤其要使用相同的 `reasoning_effort` 值，并保留选定的物理模型名称。

## 快速检查

```bash
curl -sS http://127.0.0.1:18081/healthz

curl -i http://127.0.0.1:18081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"Hello"}]}'
```

运行聚焦测试：

```bash
cd /home/chengsixiang/semantic-router
/home/chengsixiang/.venv/bin/python -m pytest \
  src/vllm-sr/tests/test_openai_record_replay_backend.py
```