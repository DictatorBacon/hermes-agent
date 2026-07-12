"""Focused tests for API server session-control endpoints."""

import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def auth_adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    return adapter


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_patch("/api/sessions/{session_id}", adapter._handle_patch_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


@pytest.mark.asyncio
async def test_capabilities_advertises_session_control_surface(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    features = data["features"]
    assert features["session_resources"] is True
    assert features["session_chat"] is True
    assert features["session_chat_streaming"] is True
    assert features["session_fork"] is True
    assert features["admin_config_rw"] is False
    assert features["memory_write_api"] is False
    assert features["skills_api"] is True
    assert features["realtime_voice"] is False
    assert data["endpoints"]["sessions"] == {"method": "GET", "path": "/api/sessions"}
    assert data["endpoints"]["session_chat_stream"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/chat/stream",
    }


@pytest.mark.asyncio
async def test_run_agent_binds_api_session_context_for_tool_env(adapter, monkeypatch):
    """API-server request sessions should reach tools and terminal subprocess env."""
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-session")
    observed = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id: str):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            from gateway.session_context import get_session_env
            from tools.environments.local import _make_run_env

            observed["task_id"] = task_id
            observed["context_session_id"] = get_session_env("HERMES_SESSION_ID")
            observed["context_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            observed["context_session_key"] = get_session_env("HERMES_SESSION_KEY")
            observed["child_session_id"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        return FakeAgent(kwargs["session_id"])

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="request-session",
        gateway_session_key="request-key",
    )

    assert result["session_id"] == "request-session"
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert observed == {
        "task_id": "request-session",
        "context_session_id": "request-session",
        "context_platform": "api_server",
        "context_session_key": "request-key",
        "child_session_id": "request-session",
    }


@pytest.mark.asyncio
async def test_session_crud_and_message_history(adapter, session_db):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        create_resp = await cli.post("/api/sessions", json={"title": "Mobile chat", "model": "test-model"})
        assert create_resp.status == 201
        created = await create_resp.json()
        session_id = created["session"]["id"]
        assert created["object"] == "hermes.session"
        assert created["session"]["title"] == "Mobile chat"

        session_db.append_message(session_id, "user", "hello from phone")
        session_db.append_message(session_id, "assistant", "hello from hermes")

        list_resp = await cli.get("/api/sessions?limit=10&offset=0")
        assert list_resp.status == 200
        listed = await list_resp.json()
        assert listed["object"] == "list"
        assert [s["id"] for s in listed["data"]] == [session_id]
        assert listed["data"][0]["message_count"] == 2

        get_resp = await cli.get(f"/api/sessions/{session_id}")
        assert get_resp.status == 200
        got = await get_resp.json()
        assert got["session"]["id"] == session_id
        assert got["session"]["message_count"] == 2

        messages_resp = await cli.get(f"/api/sessions/{session_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()
        assert messages["object"] == "list"
        assert [m["role"] for m in messages["data"]] == ["user", "assistant"]
        assert messages["data"][0]["content"] == "hello from phone"

        patch_resp = await cli.patch(f"/api/sessions/{session_id}", json={"title": "Renamed"})
        assert patch_resp.status == 200
        patched = await patch_resp.json()
        assert patched["session"]["title"] == "Renamed"

        delete_resp = await cli.delete(f"/api/sessions/{session_id}")
        assert delete_resp.status == 200
        deleted = await delete_resp.json()
        assert deleted == {"object": "hermes.session.deleted", "id": session_id, "deleted": True}
        assert session_db.get_session(session_id) is None


@pytest.mark.asyncio
async def test_session_messages_follow_compression_tip(adapter, session_db):
    source_id = session_db.create_session("source-session", "api_server")
    session_db.append_message(source_id, "user", "before compression")
    session_db.end_session(source_id, "compression")
    session_db.create_session("tip-session", "api_server", parent_session_id=source_id)
    session_db.append_message("tip-session", "user", "after compression")
    # A legacy explicit branch may lack _branched_from. Stale-root routing must
    # still stop at the compression tip rather than following an arbitrary child.
    session_db.end_session("tip-session", "branched")
    session_db.create_session(
        "legacy-branch", "api_server", parent_session_id="tip-session"
    )
    session_db.append_message("legacy-branch", "user", "branch-only turn")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        messages_resp = await cli.get(f"/api/sessions/{source_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()

    assert messages["object"] == "list"
    assert messages["session_id"] == "tip-session"
    assert [m["content"] for m in messages["data"]] == ["before compression", "after compression"]


@pytest.mark.asyncio
async def test_session_messages_paginate_newest_first_across_compression_lineage(
    adapter, session_db
):
    from agent.context_compressor import SUMMARY_PREFIX

    root_id = session_db.create_session("paged-root", "api_server")
    session_db.append_message(root_id, "user", "oldest")
    session_db.append_message(root_id, "assistant", "older reply")
    session_db.end_session(root_id, "compression")
    tip_id = session_db.create_session(
        "paged-tip", "api_server", parent_session_id=root_id
    )
    session_db.append_message(tip_id, "user", f"{SUMMARY_PREFIX}\ninternal")
    session_db.append_message(tip_id, "assistant", "older reply")
    session_db.append_message(tip_id, "user", "newer question")
    session_db.append_message(tip_id, "assistant", "newest reply")
    session_db.append_message(
        tip_id,
        "tool",
        "large tool output hidden by the chat renderer",
        tool_call_id="call-hidden",
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        latest_response = await cli.get(
            f"/api/sessions/{root_id}/messages?limit=2"
        )
        assert latest_response.status == 200
        latest = await latest_response.json()
        # Cursor boundaries are absolute in the append-only display projection,
        # so a new turn arriving after page one must not shift page two.
        session_db.append_message(tip_id, "user", "arrived after page one")
        session_db.append_message(tip_id, "assistant", "new arrival reply")
        older_response = await cli.get(
            f"/api/sessions/{root_id}/messages"
            f"?limit=2&before={latest['next_cursor']}"
        )
        assert older_response.status == 200
        older = await older_response.json()

    assert latest["session_id"] == tip_id
    assert [m["content"] for m in latest["data"]] == [
        "newer question",
        "newest reply",
    ]
    assert latest["has_more"] is True
    assert isinstance(latest["next_cursor"], str)
    assert [m["content"] for m in older["data"]] == [
        "oldest",
        "older reply",
    ]
    assert older["has_more"] is False
    assert older["next_cursor"] is None


@pytest.mark.asyncio
async def test_session_messages_do_not_let_tool_call_scaffolding_crowd_out_chat_turns(
    adapter, session_db
):
    session_id = session_db.create_session("paged-tool-run", "api_server")
    session_db.append_message(session_id, "user", "previous request")
    session_db.append_message(session_id, "assistant", "previous reply")
    session_db.append_message(session_id, "user", "current request")
    for index in range(8):
        call_id = f"call-{index}"
        session_db.append_message(
            session_id,
            "assistant",
            "",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        )
        session_db.append_message(
            session_id,
            "tool",
            "internal tool output",
            tool_call_id=call_id,
            tool_name="terminal",
        )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=3"
        )
        assert response.status == 200
        page = await response.json()

    assert [(message["role"], message["content"]) for message in page["data"]] == [
        ("user", "previous request"),
        ("assistant", "previous reply"),
        ("user", "current request"),
    ]
    assert page["has_more"] is False
    assert page["next_cursor"] is None


@pytest.mark.asyncio
async def test_session_messages_reject_invalid_pagination(adapter, session_db):
    session_id = session_db.create_session("paged-invalid", "api_server")
    app = _create_session_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        too_large = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=501"
        )
        bad_cursor = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=50&before=not-a-cursor"
        )
        oversized_cursor = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=50&before=v1:"
            + ("9" * 5000)
        )

    assert too_large.status == 400
    assert bad_cursor.status == 400
    assert oversized_cursor.status == 400


@pytest.mark.asyncio
async def test_session_messages_do_not_prepend_explicit_branch_parent(
    adapter, session_db
):
    """A branch already carries copied history, so display must not concatenate
    the parent again merely because it has parent_session_id set."""
    source_id = session_db.create_session("branch-source", "api_server")
    session_db.append_message(source_id, "user", "one copy only")
    branch_id = session_db.create_session(
        "explicit-branch",
        "api_server",
        parent_session_id=source_id,
        model_config={"_branched_from": source_id},
    )
    source_history = session_db.get_messages_as_conversation(source_id)
    session_db.replace_messages(
        branch_id,
        [{**message, "_context_snapshot": True} for message in source_history],
    )
    session_db.append_message(branch_id, "assistant", "branch-only reply")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{branch_id}/messages")
        assert response.status == 200
        payload = await response.json()

    assert payload["session_id"] == branch_id
    assert [m["content"] for m in payload["data"]] == [
        "one copy only",
        "branch-only reply",
    ]


@pytest.mark.asyncio
async def test_session_messages_hide_compaction_handoff_and_deduplicate_snapshot(
    adapter, session_db
):
    """The user-facing transcript stays continuous across rotating compaction."""
    from agent.context_compressor import SUMMARY_PREFIX

    root_id = session_db.create_session("display-root", "api_server")
    session_db.append_message(root_id, "user", "Build the feature")
    session_db.append_message(root_id, "assistant", "Working on it")
    session_db.end_session(root_id, "compression")

    tip_id = session_db.create_session(
        "display-tip", "api_server", parent_session_id=root_id
    )
    session_db.append_message(tip_id, "user", f"{SUMMARY_PREFIX}\ninternal handoff")
    # A preserved tail row copied into the compacted child must not appear twice.
    session_db.append_message(tip_id, "assistant", "Working on it")
    session_db.append_message(tip_id, "assistant", "Finished and verified")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{root_id}/messages")
        assert response.status == 200
        payload = await response.json()

    assert [m["content"] for m in payload["data"]] == [
        "Build the feature",
        "Working on it",
        "Finished and verified",
    ]


@pytest.mark.asyncio
async def test_session_messages_include_archived_turns_after_in_place_compaction(
    adapter, session_db
):
    """Model-context compaction must not erase the visible chat transcript."""
    from agent.context_compressor import SUMMARY_PREFIX

    session_id = session_db.create_session("display-in-place", "api_server")
    session_db.append_message(session_id, "user", "Original request")
    session_db.append_message(
        session_id,
        "assistant",
        "Original visible answer",
        reasoning="visible reasoning",
        timestamp=123.0,
    )
    session_db.archive_and_compact(
        session_id,
        [
            {"role": "user", "content": f"{SUMMARY_PREFIX}\ninternal handoff"},
            {"role": "assistant", "content": "Original visible answer"},
        ],
    )
    session_db.append_message(session_id, "assistant", "Continuation after compaction")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{session_id}/messages")
        assert response.status == 200
        payload = await response.json()

    assert [m["content"] for m in payload["data"]] == [
        "Original request",
        "Original visible answer",
        "Continuation after compaction",
    ]
    assert payload["data"][1]["timestamp"] == 123.0
    assert payload["data"][1]["reasoning"] == "visible reasoning"
    assert all("context_snapshot" not in message for message in payload["data"])
    assert all("active" not in message for message in payload["data"])


@pytest.mark.asyncio
async def test_session_messages_hide_legacy_todo_snapshot(adapter, session_db):
    """Pre-marker compaction TODO injections are model state, not chat turns."""
    session_id = session_db.create_session("display-legacy-todo", "api_server")
    session_db.append_message(session_id, "user", "Ok, let's proceed forward")
    session_db.append_message(session_id, "assistant", "Starting the review")
    session_db.append_message(
        session_id,
        "user",
        "[Your active task list was preserved across context compression]\n"
        "- [>] review. Review changes (in_progress)",
    )
    session_db.append_message(session_id, "assistant", "Review continued")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{session_id}/messages")
        assert response.status == 200
        payload = await response.json()

    assert [m["content"] for m in payload["data"]] == [
        "Ok, let's proceed forward",
        "Starting the review",
        "Review continued",
    ]


@pytest.mark.asyncio
async def test_session_messages_preserve_legitimate_repeated_turns(adapter, session_db):
    """Snapshot removal must not become global content deduplication."""
    from agent.context_compressor import SUMMARY_PREFIX

    session_id = session_db.create_session("display-repeats", "api_server")
    for role, content in [
        ("user", "repeat"),
        ("assistant", "ack"),
        ("user", "repeat"),
        ("assistant", "ack"),
    ]:
        session_db.append_message(session_id, role, content)
    session_db.archive_and_compact(
        session_id,
        [{"role": "user", "content": f"{SUMMARY_PREFIX}\ninternal handoff"}],
    )
    session_db.append_message(session_id, "user", "repeat")
    session_db.append_message(session_id, "assistant", "ack")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{session_id}/messages")
        assert response.status == 200
        payload = await response.json()

    assert [(m["role"], m["content"]) for m in payload["data"]] == [
        ("user", "repeat"),
        ("assistant", "ack"),
        ("user", "repeat"),
        ("assistant", "ack"),
        ("user", "repeat"),
        ("assistant", "ack"),
    ]


@pytest.mark.asyncio
async def test_session_messages_sanitize_internal_context_fences(adapter, session_db):
    session_id = session_db.create_session("display-sanitized", "api_server")
    session_db.append_message(
        session_id,
        "assistant",
        "<memory-context>PRIVATE INTERNAL MEMORY</memory-context>Visible answer",
    )
    session_db.append_message(
        session_id,
        "tool",
        "<memory-context>PRIVATE TOOL MEMORY</memory-context>Visible tool output",
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{session_id}/messages")
        assert response.status == 200
        payload = await response.json()

    rendered = "\n".join(str(message.get("content") or "") for message in payload["data"])
    assert "PRIVATE INTERNAL MEMORY" not in rendered
    assert "PRIVATE TOOL MEMORY" not in rendered
    assert "Visible answer" in rendered
    assert "Visible tool output" in rendered


@pytest.mark.asyncio
async def test_session_fork_uses_current_sessiondb_branch_primitives(adapter, session_db):
    source_id = session_db.create_session("source-session", "api_server", model="test-model")
    session_db.set_session_title(source_id, "Original")
    session_db.append_message(source_id, "user", "first path")
    session_db.append_message(source_id, "assistant", "answer")
    session_db.append_message(
        source_id,
        "assistant",
        "replay snapshot",
        tool_calls=[{"id": "call-1", "type": "function"}],
        reasoning="thinking",
        reasoning_details=[{"type": "summary", "text": "step"}],
        timestamp=123.0,
        context_snapshot=True,
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(f"/api/sessions/{source_id}/fork", json={"title": "Alternative"})
        assert resp.status == 201
        payload = await resp.json()

    fork = payload["session"]
    assert payload["object"] == "hermes.session"
    assert fork["id"] != source_id
    assert fork["parent_session_id"] == source_id
    assert fork["title"] == "Alternative"
    copied = session_db.get_messages_as_conversation(fork["id"])
    assert [m["content"] for m in copied] == [
        "first path",
        "answer",
        "replay snapshot",
    ]
    assert all(message["_context_snapshot"] is True for message in copied)
    assert copied[-1]["tool_calls"] == [{"id": "call-1", "type": "function"}]
    assert copied[-1]["reasoning_details"] == [{"type": "summary", "text": "step"}]
    assert copied[-1]["timestamp"] == 123.0
    fork_config = session_db.get_session(fork["id"])["model_config"]
    if isinstance(fork_config, str):
        fork_config = json.loads(fork_config)
    assert fork_config["_branched_from"] == source_id
    assert session_db.get_session(source_id)["end_reason"] == "branched"


@pytest.mark.asyncio
async def test_session_chat_loads_history_and_preserves_session_headers(auth_adapter, session_db):
    session_id = session_db.create_session("chat-session", "api_server")
    session_db.set_session_title(session_id, "Chat")
    session_db.append_message(session_id, "user", "earlier")
    session_db.append_message(session_id, "assistant", "prior answer")

    mock_run = AsyncMock(return_value=({"final_response": "fresh answer", "session_id": session_id}, {"total_tokens": 3}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={
                    "message": "next",
                    "system_message": "stay focused",
                    "model": "gpt-5.5",
                    "provider": "openai-codex",
                    "reasoning_effort": "xhigh",
                    "service_tier": "priority",
                },
                headers={"Authorization": "Bearer sk-test", "X-Hermes-Session-Key": "client-42"},
            )
            assert resp.status == 200
            payload = await resp.json()

    assert resp.headers["X-Hermes-Session-Id"] == session_id
    assert resp.headers["X-Hermes-Session-Key"] == "client-42"
    assert payload["object"] == "hermes.session.chat.completion"
    assert payload["session_id"] == session_id
    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["content"] == "fresh answer"
    mock_run.assert_awaited_once()
    _, kwargs = mock_run.call_args
    assert kwargs["session_id"] == session_id
    assert kwargs["gateway_session_key"] == "client-42"
    assert kwargs["ephemeral_system_prompt"] == "stay focused"
    assert kwargs["model_override"] == "gpt-5.5"
    assert kwargs["provider_override"] == "openai-codex"
    assert kwargs["reasoning_effort_override"] == "xhigh"
    assert kwargs["service_tier_override"] == "priority"
    history = kwargs["conversation_history"]
    assert len(history) == 2
    assert isinstance(history[0].pop("timestamp"), (int, float))
    assert isinstance(history[1].pop("timestamp"), (int, float))
    assert history == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "prior answer"},
    ]


@pytest.mark.asyncio
async def test_session_chat_accepts_multimodal_message(auth_adapter, session_db):
    session_id = session_db.create_session("image-session", "api_server")
    image_payload = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    expected_user_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    mock_run = AsyncMock(return_value=({"final_response": "A cat.", "session_id": session_id}, {"total_tokens": 4}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": image_payload},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert resp.status == 200, await resp.text()

    _, kwargs = mock_run.call_args
    assert kwargs["user_message"] == expected_user_message


@pytest.mark.asyncio
async def test_session_chat_stream_accepts_multimodal_message(adapter, session_db):
    session_id = session_db.create_session("image-stream-session", "api_server")
    image_payload = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    expected_user_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    captured_kwargs = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        kwargs["stream_delta_callback"]("A cat.")
        return {"final_response": "A cat.", "session_id": session_id}, {"total_tokens": 4}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": image_payload},
            )
            assert resp.status == 200, await resp.text()
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

    assert "event: assistant.completed" in body
    assert captured_kwargs["user_message"] == expected_user_message


@pytest.mark.asyncio
async def test_session_chat_stream_emits_lifecycle_events_and_keepalive_safe_shape(adapter, session_db):
    session_id = session_db.create_session("stream-session", "api_server")
    session_db.set_session_title(session_id, "Stream")

    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        kwargs["stream_delta_callback"]("Hello")
        kwargs["stream_delta_callback"](" world")
        kwargs["tool_progress_callback"]("reasoning.available", tool_name="_thinking", preview="thinking")
        return {"final_response": "Hello world", "session_id": session_id}, {"total_tokens": 2}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={
                    "message": "stream please",
                    "reasoning_effort": "low",
                    "service_tier": "normal",
                },
            )
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

    assert "event: run.started" in body
    assert "event: message.started" in body
    assert "event: assistant.delta" in body
    assert "Hello world" in body
    assert "event: tool.progress" in body
    assert "event: assistant.completed" in body
    assert "event: run.completed" in body
    assert "event: done" in body
    assert captured["reasoning_effort_override"] == "low"
    assert captured["service_tier_override"] == "normal"


@pytest.mark.asyncio
async def test_session_chat_stream_disconnect_interrupts_agent(adapter, session_db):
    """Client disconnects from session SSE should stop the active agent run."""
    session_id = session_db.create_session("disconnect-session", "api_server")
    fake_agent = MagicMock()
    write_count = {"n": 0}

    class DisconnectingStreamResponse:
        def __init__(self, *args, **kwargs):
            self.headers = kwargs.get("headers", {})

        async def prepare(self, request):
            return None

        async def write(self, payload):
            write_count["n"] += 1
            if write_count["n"] >= 3:
                raise ConnectionResetError("simulated disconnect")

    async def fake_run(**kwargs):
        kwargs["agent_ref"][0] = fake_agent
        kwargs["stream_delta_callback"]("partial response")
        await asyncio.sleep(60)
        return {"final_response": "should not complete", "session_id": session_id}, {"total_tokens": 1}

    request = MagicMock()
    request.headers = {}
    request.match_info = {"session_id": session_id}
    request.json = AsyncMock(return_value={"message": "start then disconnect"})

    import gateway.platforms.api_server as api_mod

    with patch.object(api_mod.web, "StreamResponse", DisconnectingStreamResponse):
        with patch.object(adapter, "_run_agent", side_effect=fake_run):
            await adapter._handle_session_chat_stream(request)

    fake_agent.interrupt.assert_called_once_with("SSE client disconnected")


@pytest.mark.asyncio
async def test_session_chat_stream_rejects_non_string_model(adapter, session_db):
    session_id = session_db.create_session("bad-model-session", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat/stream",
            json={"message": "hello", "model": {"bad": "shape"}},
        )
        assert resp.status == 400
        data = await resp.json()
        assert data["error"]["code"] == "invalid_model"


@pytest.mark.asyncio
async def test_session_chat_rejects_non_string_provider(adapter, session_db):
    session_id = session_db.create_session("bad-provider-session", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "hello", "provider": ["bad"]},
        )
        assert resp.status == 400
        data = await resp.json()
        assert data["error"]["code"] == "invalid_provider"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_field", "runtime_value", "error_code"),
    [
        ("reasoning_effort", "ultra", "invalid_reasoning_effort"),
        ("reasoning_effort", {"bad": "shape"}, "invalid_reasoning_effort"),
        ("service_tier", "turbo", "invalid_service_tier"),
        ("service_tier", True, "invalid_service_tier"),
    ],
)
async def test_session_chat_rejects_invalid_runtime_controls(
    adapter,
    session_db,
    runtime_field,
    runtime_value,
    error_code,
):
    session_id = session_db.create_session("bad-runtime-session", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "hello", runtime_field: runtime_value},
        )
        assert resp.status == 400
        data = await resp.json()
        assert data["error"]["code"] == error_code


@pytest.mark.asyncio
async def test_session_chat_stream_run_completed_carries_turn_transcript(adapter, session_db):
    """run.completed must include the full interleaved turn transcript so a
    client that lost intermediate (pre-tool-call) assistant text from the live
    delta stream can reconcile without a separate /messages fetch. Refs #34703.
    """
    import json as _json

    session_id = session_db.create_session("transcript-session", "api_server")

    async def fake_run(**kwargs):
        # Stream the intermediate planning text the way a real turn would.
        kwargs["stream_delta_callback"]("Let me search for that:")
        kwargs["stream_delta_callback"]("Here is the summary.")
        result = {
            "final_response": "Here is the summary.",
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "search then summarize"},
                {
                    "role": "assistant",
                    "content": "Let me search for that:",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "results", "tool_call_id": "call_1", "tool_name": "web_search"},
                {"role": "assistant", "content": "Here is the summary."},
            ],
        }
        return result, {"total_tokens": 6}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "search then summarize"},
            )
            assert resp.status == 200
            body = await resp.text()

    # Pull the run.completed event payload out of the SSE body.
    run_completed_payload = None
    for block in body.split("\n\n"):
        if "event: run.completed" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    run_completed_payload = _json.loads(line[len("data: "):])
            break
    assert run_completed_payload is not None, body
    messages = run_completed_payload.get("messages")
    assert isinstance(messages, list) and messages, run_completed_payload

    # The colon-ended intermediate text that preceded the tool call must be present.
    contents = [m.get("content") for m in messages]
    assert "Let me search for that:" in contents
    assert "Here is the summary." in contents
    # No prior-turn user message should leak into the per-turn slice.
    assert all(m.get("role") in ("assistant", "tool") for m in messages)
    # The tool call is preserved alongside the intermediate text.
    assert any(m.get("tool_calls") for m in messages)


@pytest.mark.asyncio
async def test_rotated_stream_completion_excludes_compacted_replay_snapshot(adapter, session_db):
    """A mid-turn rotation must return only this turn's visible transcript."""
    import json as _json

    root_id = session_db.create_session("rotation-root", "api_server")
    session_db.append_message(root_id, "user", "old question")
    session_db.append_message(root_id, "assistant", "old answer")

    async def fake_run(**kwargs):
        session_db.end_session(root_id, "compression")
        child_id = session_db.create_session(
            "rotation-child", "api_server", parent_session_id=root_id
        )
        session_db.append_message(
            child_id,
            "assistant",
            "synthetic compacted summary",
            context_snapshot=True,
        )
        session_db.append_message(child_id, "user", "new question")
        session_db.append_message(child_id, "assistant", "new answer")
        kwargs["stream_delta_callback"]("new answer")
        return {
            "final_response": "new answer",
            "session_id": child_id,
            "messages": [
                {"role": "assistant", "content": "synthetic compacted summary"},
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ],
        }, {"total_tokens": 3}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{root_id}/chat/stream",
                json={"message": "new question"},
            )
            assert resp.status == 200
            body = await resp.text()

    payload = None
    for block in body.split("\n\n"):
        if "event: run.completed" not in block:
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                payload = _json.loads(line[len("data: "):])
        break

    assert payload is not None, body
    assert [
        (message.get("role"), message.get("content"))
        for message in payload.get("messages", [])
    ] == [("assistant", "new answer")]


@pytest.mark.asyncio
async def test_same_session_streams_do_not_mix_completed_transcripts(adapter, session_db):
    """Concurrent requests for one session must be serialized per session."""
    session_id = session_db.create_session("concurrent-stream", "api_server")
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def fake_run(**kwargs):
        message = kwargs["user_message"]
        session_db.append_message(session_id, "user", message)
        if message == "first":
            first_started.set()
            await release_first.wait()
        answer = f"answer {message}"
        session_db.append_message(session_id, "assistant", answer)
        kwargs["stream_delta_callback"](answer)
        return {
            "final_response": answer,
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ],
        }, {"total_tokens": 2}

    def completed_contents(body):
        for block in body.split("\n\n"):
            if "event: run.completed" not in block:
                continue
            for line in block.splitlines():
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: "):])
                    return [m.get("content") for m in payload.get("messages", [])]
        raise AssertionError(body)

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            first_task = asyncio.create_task(
                cli.post(
                    f"/api/sessions/{session_id}/chat/stream",
                    json={"message": "first"},
                )
            )
            await first_started.wait()
            second_task = asyncio.create_task(
                cli.post(
                    f"/api/sessions/{session_id}/chat/stream",
                    json={"message": "second"},
                )
            )
            await asyncio.sleep(0.05)
            release_first.set()
            first_resp, second_resp = await asyncio.gather(first_task, second_task)
            first_body, second_body = await asyncio.gather(
                first_resp.text(), second_resp.text()
            )

    assert completed_contents(first_body) == ["answer first"]
    assert completed_contents(second_body) == ["answer second"]


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_finish_before_worker(adapter):
    """Cancellation must not release session ownership while a worker still runs."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def worker():
        started.set()
        await release.wait()
        return "done"

    waiter = asyncio.create_task(adapter._await_uncancellable(worker()))
    await started.wait()
    waiter.cancel()
    await asyncio.sleep(0.05)
    assert not waiter.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter


@pytest.mark.asyncio
async def test_queued_stream_re_resolves_rotated_lineage_tip(adapter, session_db):
    root_id = session_db.create_session("queued-root", "api_server")
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    seen = {}

    async def fake_run(**kwargs):
        message = kwargs["user_message"]
        if message == "first":
            first_started.set()
            await release_first.wait()
            session_db.end_session(root_id, "compression")
            child_id = session_db.create_session(
                "queued-child", "api_server", parent_session_id=root_id
            )
            return {"final_response": "first done", "session_id": child_id}, {}
        seen["second_session_id"] = kwargs["session_id"]
        return {"final_response": "second done", "session_id": kwargs["session_id"]}, {}

    async def post_and_read(cli, message):
        response = await cli.post(
            f"/api/sessions/{root_id}/chat/stream", json={"message": message}
        )
        return await response.text()

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            first = asyncio.create_task(post_and_read(cli, "first"))
            await first_started.wait()
            second = asyncio.create_task(post_and_read(cli, "second"))
            await asyncio.sleep(0.05)
            release_first.set()
            await asyncio.gather(first, second)

    assert seen["second_session_id"] == "queued-child"


@pytest.mark.asyncio
async def test_queued_nonstream_chat_re_resolves_rotated_lineage_tip(adapter, session_db):
    root_id = session_db.create_session("queued-chat-root", "api_server")
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    seen = {}

    async def fake_run(**kwargs):
        message = kwargs["user_message"]
        if message == "first":
            first_started.set()
            await release_first.wait()
            session_db.end_session(root_id, "compression")
            child_id = session_db.create_session(
                "queued-chat-child", "api_server", parent_session_id=root_id
            )
            return {"final_response": "first done", "session_id": child_id}, {}
        seen["second_session_id"] = kwargs["session_id"]
        return {"final_response": "second done", "session_id": kwargs["session_id"]}, {}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            first = asyncio.create_task(
                cli.post(f"/api/sessions/{root_id}/chat", json={"message": "first"})
            )
            await first_started.wait()
            second = asyncio.create_task(
                cli.post(f"/api/sessions/{root_id}/chat", json={"message": "second"})
            )
            await asyncio.sleep(0.05)
            release_first.set()
            await asyncio.gather(first, second)

    assert seen["second_session_id"] == "queued-chat-child"


@pytest.mark.asyncio
async def test_in_place_compaction_stream_reports_only_current_turn(adapter, session_db):
    """Archival metadata changes must not trigger stale transcript fallback."""
    session_id = session_db.create_session("in-place-stream", "api_server")
    session_db.append_message(session_id, "user", "old question")
    session_db.append_message(session_id, "assistant", "old answer")

    async def fake_run(**kwargs):
        session_db.archive_and_compact(
            session_id,
            [
                {
                    "role": "user",
                    "content": "[CONTEXT COMPACTION] old summary",
                    "_context_snapshot": True,
                }
            ],
        )
        session_db.append_message(session_id, "user", kwargs["user_message"])
        session_db.append_message(session_id, "assistant", "new answer")
        return {
            "final_response": "new answer",
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": kwargs["user_message"]},
                {"role": "assistant", "content": "new answer"},
            ],
        }, {"total_tokens": 2}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "new question"},
            )
            body = await response.text()

    completed = next(
        json.loads(line[len("data: "):])
        for block in body.split("\n\n")
        if "event: run.completed" in block
        for line in block.splitlines()
        if line.startswith("data: ")
    )
    assert [message["content"] for message in completed["messages"]] == [
        "new answer"
    ]


@pytest.mark.asyncio
async def test_session_chat_stream_resolves_stale_compression_root(adapter, session_db):
    """If the client sends to a compression-ended root, the API must resolve
    to the current tip before running, preventing sibling continuations."""
    root_id = session_db.create_session("root-session", "api_server")
    session_db.append_message(root_id, "user", "hello root")
    session_db.end_session(root_id, "compression")
    tip_id = session_db.create_session("tip-session", "api_server", parent_session_id=root_id)
    session_db.append_message(tip_id, "user", "hello tip")
    session_db.end_session(tip_id, "branched")
    session_db.create_session(
        "legacy-stream-branch", "api_server", parent_session_id=tip_id
    )
    session_db.append_message("legacy-stream-branch", "user", "branch only")

    captured_kwargs = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        kwargs["stream_delta_callback"]("response")
        return {"final_response": "response", "session_id": tip_id}, {"total_tokens": 1}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{root_id}/chat/stream",
                json={"message": "next message"},
            )
            assert resp.status == 200

    assert captured_kwargs["session_id"] == tip_id, (
        "Chat stream must resolve a stale compression root to the current tip"
    )


@pytest.mark.asyncio
async def test_session_chat_resolves_stale_compression_root(adapter, session_db):
    """Non-streaming chat must also resolve a stale root to the tip."""
    root_id = session_db.create_session("root-chat", "api_server")
    session_db.append_message(root_id, "user", "hello root")
    session_db.end_session(root_id, "compression")
    tip_id = session_db.create_session("tip-chat", "api_server", parent_session_id=root_id)
    session_db.append_message(tip_id, "user", "hello tip")
    session_db.end_session(tip_id, "branched")
    session_db.create_session(
        "legacy-chat-branch", "api_server", parent_session_id=tip_id
    )
    session_db.append_message("legacy-chat-branch", "user", "branch only")

    mock_run = AsyncMock(return_value=({"final_response": "ok", "session_id": tip_id}, {"total_tokens": 1}))
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{root_id}/chat",
                json={"message": "next"},
            )
            assert resp.status == 200

    _, kwargs = mock_run.call_args
    assert kwargs["session_id"] == tip_id


@pytest.mark.asyncio
async def test_session_endpoints_require_auth_when_key_configured(auth_adapter):
    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/sessions")
        assert resp.status == 401
        body = await resp.json()
        assert body["error"]["code"] == "invalid_api_key"

        ok = await cli.get("/api/sessions", headers={"Authorization": "Bearer sk-test"})
        assert ok.status == 200
        data = await ok.json()
        assert data["object"] == "list"
        assert data["data"] == []


@pytest.mark.asyncio
async def test_session_header_rejected_without_api_key(adapter, session_db):
    session_id = session_db.create_session("unsafe-session", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "hello"},
            headers={"X-Hermes-Session-Key": "client-42"},
        )
        assert resp.status == 403
        data = await resp.json()
        assert "X-Hermes-Session-Key requires API key" in data["error"]["message"]
