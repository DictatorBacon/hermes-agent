"""Tests for in-place context compaction (config: compression.in_place, #38763).

When ``compression.in_place`` is True, ``compress_context()`` rewrites the
message list and rebuilds the system prompt but keeps the SAME ``session_id``:
no ``end_session``, no ``parent_session_id`` child row, no ``name #N`` title
renumber, no flush-cursor reset. This eliminates the session-rotation bug
cluster (#33618 /goal loss, #14238 lost response, #33907 orphans, #45117 search
gaps, #42228 null cwd). When the flag is False (default), rotation behaves
exactly as before.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_agent(session_db, session_id, *, in_place):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = in_place
    # Mock the compressor to return a deterministic shrunk transcript so the
    # test exercises the DB-mutation path, not summarization quality.
    def _fake_compress(messages, current_tokens=None, focus_topic=None, force=False):
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
            {"role": "assistant", "content": "recent reply"},
        ]

    agent.context_compressor.compress = _fake_compress
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor.compression_count = 1
    return agent


def _seed(db, sid, title, n=8):
    db.create_session(sid, "cli", model="test/model")
    db.set_session_title(sid, title)
    messages = []
    for i in range(n):
        message = {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"msg {i}",
            "_db_persisted": True,
        }
        messages.append(message)
        db.append_message(
            session_id=sid,
            role=message["role"],
            content=message["content"],
        )
    return messages


class TestInPlaceCompaction:
    def test_in_place_keeps_same_session_id(self):
        """In-place mode: id unchanged, no child row, no rename, history kept."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_120000_aaaaaa"
            messages = _seed(db, sid, "my-research")
            agent = _make_agent(db, sid, in_place=True)
            agent._last_flushed_db_idx = 5

            compressed, _sp = compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )

            # Identity never moved.
            assert agent.session_id == sid
            # No continuation row forked.
            child = db._conn.execute(
                "SELECT id FROM sessions WHERE parent_session_id = ?", (sid,)
            ).fetchall()
            assert child == []
            # Session not ended; title untouched (no "#2").
            row = db.get_session(sid)
            assert row["end_reason"] is None
            assert row["title"] == "my-research"
            # DURABLE, NON-DESTRUCTIVE compaction (the core invariant, per
            # Teknium's review): the LIVE context is the compacted set, but the
            # pre-compaction turns are PRESERVED on disk (active=0), not deleted
            # — searchable + recoverable under the SAME id. A resume reloads the
            # compacted set so compaction actually shrinks the live session and
            # doesn't immediately re-compact (#38763).
            reloaded = db.get_messages_as_conversation(sid)
            assert len(reloaded) == 2
            assert [m.get("content") for m in reloaded] == [
                "[CONTEXT COMPACTION] summary of prior turns",
                "recent reply",
            ]
            assert row["message_count"] == 2  # live (active) count
            # NON-DESTRUCTIVE: the 8 seeded originals survive at active=0
            # alongside the 2 compacted rows — nothing was DELETEd.
            all_rows = db.get_messages(sid, include_inactive=True)
            assert len(all_rows) == 10
            archived = [m for m in all_rows if not m.get("active", 1)]
            assert len(archived) == 8
            # The originals remain FTS-searchable (active=0 is a content-
            # preserving UPDATE; the fts triggers don't key on active).
            hit = db._conn.execute(
                "SELECT 1 FROM messages_fts f JOIN messages m ON m.id = f.rowid "
                "WHERE m.session_id = ? AND messages_fts MATCH 'msg' AND m.active = 0 "
                "LIMIT 1",
                (sid,),
            ).fetchone()
            assert hit is not None
            # Flush identity/cursor reset so next-turn appends diff against the
            # compacted transcript (rebuilds the identity set on next flush).
            assert agent._last_flushed_db_idx == 0
            assert agent._flushed_db_message_ids == set()
            # Rotation-independent in-place signal set for the gateway.
            assert agent._last_compaction_in_place is True
            # Live transcript actually shrank.
            assert len(compressed) == 2
            db.close()

    def test_in_place_alternation_preserved(self):
        """The compacted list must not introduce consecutive same-role messages."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_120500_cccccc"
            messages = _seed(db, sid, "alt")
            agent = _make_agent(db, sid, in_place=True)
            compressed, _ = compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )
            roles = [m["role"] for m in compressed if m.get("role") != "system"]
            assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
            db.close()

    def test_in_place_preflushes_pending_messages(self):
        """In-place compaction must durably flush the current turn before the
        compacted replay snapshot is hidden from the visible transcript."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            _seed(db, "ip_flush", "f")
            agent = _make_agent(db, "ip_flush", in_place=True)
            calls = {"n": 0}
            agent._flush_messages_to_session_db = lambda *a, **k: calls.__setitem__(
                "n", calls["n"] + 1
            )
            compress_context(
                agent, [{"role": "user", "content": "x"}] * 8,
                approx_tokens=100_000, system_message="sys",
            )
            db.close()
            assert calls["n"] == 1

    def test_in_place_preflush_keeps_pending_tool_result_visible(self):
        """A just-produced tool result must survive as durable visible history.

        The compressor's returned rows are model-context snapshots and therefore
        hidden from display. Without the preflush, a tool result that only exists
        in memory disappears from the human transcript during in-place archival.
        """
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "ip_tool_result"
            db.create_session(sid, "cli", model="test/model")
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "date", "arguments": "{}"},
                }
            ]
            db.append_message(sid, "user", "What time is it?")
            db.append_message(
                sid,
                "assistant",
                "",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
            )
            messages = [
                {
                    "role": "user",
                    "content": "What time is it?",
                    "_db_persisted": True,
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                    "finish_reason": "tool_calls",
                    "_db_persisted": True,
                },
                {
                    "role": "tool",
                    "content": "Thursday, July 9, 2026",
                    "tool_call_id": "call_1",
                    "tool_name": "date",
                },
            ]
            agent = _make_agent(db, sid, in_place=True)

            compress_context(
                agent,
                messages,
                approx_tokens=100_000,
                system_message="sys",
            )

            visible = db.get_messages_for_display(sid)
            db.close()
            assert [(m["role"], m.get("content")) for m in visible] == [
                ("user", "What time is it?"),
                ("assistant", ""),
                ("tool", "Thursday, July 9, 2026"),
            ]

    def test_display_transcript_preserves_message_content_bytes(self):
        """Display retrieval must not trim Markdown-significant whitespace.

        The browser renders streamed content before reloading canonical history.
        If this getter strips the persisted bytes, the same response visibly
        changes after the post-stream reload.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = db.create_session("display-bytes", "api_server")
            content = "    indented markdown\n\n```python\n    x = 1\n```\n\n"
            db.append_message(sid, "assistant", content)

            visible = db.get_messages_for_display(sid)
            db.close()

        assert visible[0]["content"] == content

    def test_display_keeps_legitimate_whitespace_prefixed_legacy_marker(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = db.create_session("display-marker-quote", "api_server")
            content = (
                "  [Your active task list was preserved across context compression]\n"
                "This is user-authored quoted text, not a replay snapshot."
            )
            db.append_message(sid, "user", content)
            visible = db.get_messages_for_display(sid)
            db.close()

        assert [message["content"] for message in visible] == [content]

    def test_display_keeps_legitimate_exact_legacy_marker(self):
        """Ambiguous legacy text must stay visible when provenance is unknown."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = db.create_session("display-exact-marker", "api_server")
            content = (
                "[Your active task list was preserved across context compression]\n"
                "This is a genuine user message."
            )
            db.append_message(sid, "user", content)
            visible = db.get_messages_for_display(sid)
            db.close()

        assert [message["content"] for message in visible] == [content]

    def test_prompt_rebuild_observes_memory_committed_at_boundary(self):
        """The continuation prompt must include memory produced by compaction."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "memory-prompt-boundary"
            messages = _seed(db, sid, "memory")
            agent = _make_agent(db, sid, in_place=True)
            state = {"memory": "before"}
            agent._cached_system_prompt = "stable prompt prefix"
            agent._build_system_prompt = MagicMock(
                side_effect=lambda _message: f"prompt with memory-{state['memory']}"
            )

            def commit_memory(_messages):
                state["memory"] = "after"

            agent.commit_memory_session = commit_memory

            _compressed, system_prompt = compress_context(
                agent,
                messages,
                approx_tokens=100_000,
                system_message="sys",
            )
            persisted = db.get_session(sid)["system_prompt"]
            db.close()

        assert system_prompt == "prompt with memory-after"
        assert agent._cached_system_prompt == "prompt with memory-after"
        assert persisted == "prompt with memory-after"

    @pytest.mark.parametrize("in_place", [True, False])
    def test_failed_preflush_preserves_prompt_cache(self, in_place):
        """Cancelled compaction is a strict no-op for the prompt cache prefix."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = db.create_session("flush-failure", "api_server")
            messages = [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
            agent = _make_agent(db, sid, in_place=in_place)
            agent._cached_system_prompt = "stable prompt prefix"
            agent._flush_messages_to_session_db = MagicMock(return_value=False)
            agent._invalidate_system_prompt = MagicMock()
            agent._build_system_prompt = MagicMock(return_value="rebuilt prompt")

            result, system_prompt = compress_context(
                agent,
                messages,
                approx_tokens=100_000,
                system_message="sys",
            )
            db.close()

        assert result is messages
        assert system_prompt == "stable prompt prefix"
        assert agent._cached_system_prompt == "stable prompt prefix"
        agent._invalidate_system_prompt.assert_not_called()

    @pytest.mark.parametrize("in_place", [True, False])
    def test_prompt_failure_cannot_commit_compaction_boundary(self, in_place):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "prompt-failure"
            messages = _seed(db, sid, "atomic")
            agent = _make_agent(db, sid, in_place=in_place)
            agent._cached_system_prompt = "stable prompt prefix"
            agent._flush_messages_to_session_db = MagicMock(return_value=True)
            agent.commit_memory_session = MagicMock()
            agent._build_system_prompt = MagicMock(
                side_effect=RuntimeError("prompt rebuild failed")
            )

            result, system_prompt = compress_context(
                agent,
                messages,
                approx_tokens=100_000,
                system_message="sys",
            )
            rows = db.get_messages(sid, include_inactive=True)
            children = db._conn.execute(
                "SELECT id FROM sessions WHERE parent_session_id = ?", (sid,)
            ).fetchall()
            root = db.get_session(sid)
            db.close()

        assert result is messages
        assert system_prompt == "stable prompt prefix"
        assert all(row["active"] and not row["compacted"] for row in rows)
        assert children == []
        assert root["end_reason"] is None
        assert agent.session_id == sid
        agent.commit_memory_session.assert_not_called()

    def test_compressor_copy_without_summary_window_is_strict_noop(self):
        from agent.context_compressor import ContextCompressor

        with patch(
            "agent.context_compressor.get_model_context_length", return_value=100_000
        ):
            compressor = ContextCompressor(model="test/model", quiet_mode=True)
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
            for i in range(10)
        ]
        compressor._prune_old_tool_results = MagicMock(
            return_value=(list(messages), 1)
        )
        compressor._protect_head_size = MagicMock(return_value=3)
        compressor._align_boundary_forward = MagicMock(return_value=3)
        compressor._protect_active_user_boundary = MagicMock(return_value=3)
        compressor._find_tail_cut_by_tokens = MagicMock(return_value=3)

        result = compressor.compress(messages, current_tokens=100_000)

        assert result is messages

    def test_rotation_still_preflushes(self):
        """Rotation MUST pre-flush so current-turn messages survive in the
        preserved old (parent) session before it is ended (#47202)."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            _seed(db, "rot_flush", "f")
            agent = _make_agent(db, "rot_flush", in_place=False)
            calls = {"n": 0}
            agent._flush_messages_to_session_db = lambda *a, **k: calls.__setitem__(
                "n", calls["n"] + 1
            )
            compress_context(
                agent, [{"role": "user", "content": "x"}] * 8,
                approx_tokens=100_000, system_message="sys",
            )
            assert calls["n"] == 1
            db.close()


class TestRotationFallbackWhenFlagOff:
    def test_rotation_when_flag_off(self):
        """Rotation is now the OPT-OUT fallback (default flipped to in-place in
        #38763). With in_place=False explicitly set, legacy rotation is
        unchanged — forks a renamed continuation session."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_130000_bbbbbb"
            messages = _seed(db, sid, "my-research")
            agent = _make_agent(db, sid, in_place=False)
            agent._last_flushed_db_idx = 5

            compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )

            # Identity rotated to a fresh id.
            assert agent.session_id != sid
            # Old session ended via compression; continuation forked + renamed.
            assert db.get_session(sid)["end_reason"] == "compression"
            child = db._conn.execute(
                "SELECT id, title FROM sessions WHERE parent_session_id = ?", (sid,)
            ).fetchall()
            assert len(child) == 1
            assert child[0]["title"] == "my-research #2"
            # Flush cursor reset for the new row.
            assert agent._last_flushed_db_idx == 0
            # Rotation mode does NOT set the in-place signal.
            assert getattr(agent, "_last_compaction_in_place", False) is False
            db.close()


class TestInPlaceSignalForGateway:
    """compress_context must expose a rotation-independent flag the gateway can
    read (instead of an id-change diff) to re-baseline transcript handling."""

    def test_signal_set_on_in_place_unset_on_rotation(self):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            # in-place → flag True
            ip_messages = _seed(db, "s_ip", "ip")
            a_ip = _make_agent(db, "s_ip", in_place=True)
            compress_context(
                a_ip, ip_messages,
                approx_tokens=100_000, system_message="sys",
            )
            assert a_ip._last_compaction_in_place is True

            # rotation → flag False
            rot_messages = _seed(db, "s_rot", "rot")
            a_rot = _make_agent(db, "s_rot", in_place=False)
            compress_context(
                a_rot, rot_messages,
                approx_tokens=100_000, system_message="sys",
            )
            assert a_rot._last_compaction_in_place is False
            db.close()


class TestInPlaceConfigDefault:
    def test_flag_defaults_on(self):
        """In-place is the default as of #38763 (rotation is now opt-out via
        compression.in_place: false)."""
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["compression"].get("in_place") is True


class TestCompactedTurnsStaySearchable:
    """Teknium's review hinges on the pre-compaction transcript staying
    DISCOVERABLE after in-place compaction. Compaction-archived rows
    (active=0, compacted=1) must surface in session_search by default, while
    rewind/undo rows (active=0, compacted=0) must stay hidden. The two share
    the active flag but are distinguished by the compacted flag."""

    def test_compacted_turns_found_by_default_search(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_search"
            db.create_session(sid, "cli", model="test/model")
            for r, c in [
                ("user", "configure the HMAC secret"),
                ("assistant", "set it in config.yaml"),
                ("user", "deploy returns 403"),
                ("assistant", "rotate the HMAC"),
                ("user", "works now"),
                ("assistant", "great"),
            ]:
                db.append_message(session_id=sid, role=r, content=c)

            before = db.search_messages("HMAC", role_filter=["user", "assistant"])
            assert len(before) == 2

            db.archive_and_compact(
                sid,
                [
                    {"role": "user", "content": "[SUMMARY] earlier setup"},
                    {"role": "assistant", "content": "ok"},
                ],
            )

            # The archived originals (active=0, compacted=1) are still found by
            # the DEFAULT search — this is the durability requirement.
            after = db.search_messages("HMAC", role_filter=["user", "assistant"])
            assert {m["id"] for m in after} == {1, 4}
            # Live context still excludes them.
            assert len(db.get_messages_as_conversation(sid)) == 2
            db.close()

    def test_rewound_turns_stay_hidden(self):
        """Rewind/undo (active=0, compacted=0) must NOT leak into default
        search — the distinction the compacted flag preserves."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_undo"
            db.create_session(sid, "cli", model="test/model")
            db.append_message(session_id=sid, role="user", content="ZEBRAWORD remember this")
            db.append_message(session_id=sid, role="assistant", content="noted")
            db.rewind_to_message(sid, db.get_messages(sid)[0]["id"])

            assert db.search_messages("ZEBRAWORD", role_filter=["user", "assistant"]) == []
            recovered = db.search_messages(
                "ZEBRAWORD", role_filter=["user", "assistant"], include_inactive=True
            )
            assert len(recovered) == 1
            db.close()

