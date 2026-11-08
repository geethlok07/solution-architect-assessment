"""Incremental Server-Sent Events parsing and encoding.

Network reads do not respect event, line or even character boundaries, so
the parser is fed raw bytes and keeps two pieces of state: an incremental
UTF-8 decoder (a multi-byte character split across reads is reassembled,
never replaced with U+FFFD) and the current partial line.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, field

_LINE_BREAK = re.compile(r"\r\n|\r|\n")


@dataclass
class SSEEvent:
    data: str = ""
    event: str | None = None
    id: str | None = None
    retry: int | None = None
    comments: list[str] = field(default_factory=list)

    def encode(self) -> bytes:
        lines = [f":{comment}" for comment in self.comments]
        if self.event is not None:
            lines.append(f"event: {self.event}")
        if self.id is not None:
            lines.append(f"id: {self.id}")
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")
        if self.data or not lines:
            lines.extend(f"data: {line}" for line in self.data.split("\n"))
        return ("\n".join(lines) + "\n\n").encode("utf-8")


class SSEParser:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._line = ""
        self._pending_cr = False
        self._reset_event()

    def _reset_event(self) -> None:
        self._data: list[str] = []
        self._event: str | None = None
        self._id: str | None = None
        self._retry: int | None = None
        self._comments: list[str] = []

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        return self._consume(self._decoder.decode(chunk))

    def close(self) -> list[SSEEvent]:
        """Dispatch whatever is complete at end of stream."""
        events = self._consume(self._decoder.decode(b"", final=True))
        if self._line:
            self._process_line(self._line)
            self._line = ""
        dispatched = self._dispatch()
        return [*events, dispatched] if dispatched else events

    def _consume(self, text: str) -> list[SSEEvent]:
        if self._pending_cr and text.startswith("\n"):
            text = text[1:]  # second half of a CRLF split across reads
        self._pending_cr = False
        if not text:
            return []

        buffer = self._line + text
        lines = _LINE_BREAK.split(buffer)
        self._line = lines.pop()  # incomplete trailing line, if any
        self._pending_cr = buffer.endswith("\r")

        events: list[SSEEvent] = []
        for line in lines:
            if line:
                self._process_line(line)
            elif (event := self._dispatch()) is not None:
                events.append(event)
        return events

    def _process_line(self, line: str) -> None:
        if line.startswith(":"):
            self._comments.append(line[1:])
            return
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        elif name == "id" and "\0" not in value:
            self._id = value
        elif name == "retry" and value.isdigit():
            self._retry = int(value)

    def _dispatch(self) -> SSEEvent | None:
        if not (self._data or self._event or self._id or self._retry is not None or self._comments):
            return None
        event = SSEEvent(
            data="\n".join(self._data),
            event=self._event,
            id=self._id,
            retry=self._retry,
            comments=self._comments,
        )
        self._reset_event()
        return event
