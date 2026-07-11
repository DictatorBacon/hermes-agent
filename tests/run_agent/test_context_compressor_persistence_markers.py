from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    _fresh_compaction_message_copy,
    _strip_persistence_markers,
)


def test_fresh_compaction_copy_removes_db_marker_without_mutating_source():
    source = {
        "role": "user",
        "content": "continue",
        "_db_persisted": True,
        "_context_snapshot": True,
    }

    copied = _fresh_compaction_message_copy(source)

    assert copied == {
        "role": "user",
        "content": "continue",
        "_context_snapshot": True,
    }
    assert source["_db_persisted"] is True


def test_terminal_marker_sweep_removes_markers_from_all_compacted_messages():
    messages = [
        {"role": "user", "content": "one", "_db_persisted": True},
        {"role": "assistant", "content": "two", "_db_persisted": False},
        {"role": "user", "content": "three"},
    ]

    _strip_persistence_markers(messages)

    assert all("_db_persisted" not in message for message in messages)
    assert [message["content"] for message in messages] == ["one", "two", "three"]


def test_compress_strips_db_markers_without_mutating_persisted_source_messages():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    source = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message-{index}",
            "_db_persisted": True,
        }
        for index in range(10)
    ]

    with patch(
        "agent.context_compressor.call_llm",
        side_effect=RuntimeError("provider unavailable"),
    ):
        compressed = compressor.compress(source)

    assert len(compressed) < len(source)
    assert all("_db_persisted" not in message for message in compressed)
    assert all(message["_db_persisted"] is True for message in source)
