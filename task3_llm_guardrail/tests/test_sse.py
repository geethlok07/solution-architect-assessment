from __future__ import annotations

import random

from task3_llm_guardrail.sse import SSEEvent, SSEParser

STREAM = (
    'event: message\r\ndata: {"text":"héllo 😀 世界"}\r\n\r\n'
    ": keep-alive\n\n"
    "id: 7\nretry: 1500\ndata: line one\ndata: line two\n\n"
    "data: [DONE]\n\n"
).encode()

EXPECTED = [
    ("message", None, None, '{"text":"héllo 😀 世界"}', []),
    (None, None, None, "", [" keep-alive"]),
    (None, "7", 1500, "line one\nline two", []),
    (None, None, None, "[DONE]", []),
]


def _summary(events: list[SSEEvent]) -> list[tuple]:
    return [(e.event, e.id, e.retry, e.data, e.comments) for e in events]


def test_parses_complete_stream() -> None:
    parser = SSEParser()
    assert _summary(parser.feed(STREAM) + parser.close()) == EXPECTED


def test_any_byte_level_split_gives_the_same_events() -> None:
    for seed in range(300):
        rng = random.Random(seed)
        parser, events, index = SSEParser(), [], 0
        while index < len(STREAM):
            size = rng.randint(1, 7)
            events += parser.feed(STREAM[index : index + size])
            index += size
        assert _summary(events + parser.close()) == EXPECTED, seed


def test_crlf_split_between_reads_is_one_line_break() -> None:
    parser = SSEParser()
    events = parser.feed(b"data: a\r") + parser.feed(b"\n\r") + parser.feed(b"\ndata: b\n\n")
    assert [e.data for e in events] == ["a", "b"]


def test_multibyte_character_split_across_reads() -> None:
    encoded = "data: 😀\n\n".encode()
    parser = SSEParser()
    events = [e for i in range(len(encoded)) for e in parser.feed(encoded[i : i + 1])]
    assert [e.data for e in events] == ["😀"]


def test_trailing_event_without_blank_line_is_dispatched_on_close() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: tail") == []
    assert [e.data for e in parser.close()] == ["tail"]


def test_encode_round_trips() -> None:
    event = SSEEvent(data="a\nb", event="message", id="3", comments=["ping"])
    parser = SSEParser()
    (parsed,) = parser.feed(event.encode())
    assert (parsed.data, parsed.event, parsed.id, parsed.comments) == ("a\nb", "message", "3", ["ping"])
