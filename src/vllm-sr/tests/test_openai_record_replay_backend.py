import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "openai_record_replay_backend.py"
SPEC = importlib.util.spec_from_file_location("openai_record_replay_backend", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_cache_key_is_stable_and_model_sensitive():
    payload = {"model": "small", "messages": [{"role": "user", "content": "hello"}]}
    reordered = {"messages": [{"content": "hello", "role": "user"}], "model": "small"}

    assert replay.cache_key(payload) == replay.cache_key(reordered)
    assert replay.cache_key(payload) != replay.cache_key({**payload, "model": "large"})


def test_store_round_trip(tmp_path):
    store = replay.ReplayStore(tmp_path)
    payload = {"model": "selected-model", "messages": [{"role": "user", "content": "prompt"}]}
    record = {"status_code": 200, "content_type": "application/json", "body": '{"choices": []}'}

    key = store.put(payload, record)

    assert len(key) == 64
    assert store.get(payload)["body"] == record["body"]
    assert store.get({**payload, "model": "another-model"}) is None


def test_router_rewritten_chat_path_is_supported():
    assert "/chat/completions" in replay.CHAT_COMPLETION_PATHS
    assert "/v1/chat/completions" in replay.CHAT_COMPLETION_PATHS


def test_upstream_session_ignores_environment_proxy_settings(monkeypatch):
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(name, "http://proxy.invalid:8080")

    session = replay.new_upstream_session()
    try:
        assert session.trust_env is False
        settings = session.merge_environment_settings(
            "https://upstream.invalid/chat/completions",
            proxies=None,
            stream=False,
            verify=True,
            cert=None,
        )
        assert settings["proxies"] == {}
    finally:
        session.close()


def test_explicit_upstream_proxy_remains_opt_in(tmp_path):
    server = replay.ReplayHTTPServer(
        ("127.0.0.1", 0),
        replay.ReplayStore(tmp_path),
        "auto",
        "https://upstream.invalid/v1",
        "",
        10,
        "http://proxy.example:8080",
    )
    try:
        assert server.proxies == {
            "http": "http://proxy.example:8080",
            "https": "http://proxy.example:8080",
        }
    finally:
        server.server_close()


def test_import_routerarena_detail_uses_prompt_and_selected_model(tmp_path):
    detail = tmp_path / "detail.jsonl"
    detail.write_text(
        json.dumps(
            {
                "http_status": 200,
                "selected_model": "physical-model",
                "prompt": "test prompt",
                "http_elapsed_ms": 123.4,
                "raw_response": {"model": "physical-model", "choices": [{"message": {"content": "answer"}}]},
                "task_score": 1.0,
                "is_supported": True,
                "global_index": "example_1",
            }
        )
        + "\n"
        + json.dumps({"http_status": 500})
        + "\n",
        encoding="utf-8",
    )
    store = replay.ReplayStore(tmp_path / "cache")

    imported, skipped = replay.import_routerarena_detail(store, detail, "")
    payload = {"model": "physical-model", "messages": [{"role": "user", "content": "test prompt"}]}

    assert (imported, skipped) == (1, 1)
    assert json.loads(store.get(payload)["body"])["choices"][0]["message"]["content"] == "answer"
