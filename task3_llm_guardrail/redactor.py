"""PII redaction that works on a token stream without buffering the response.

The difficulty with streaming is that sensitive values arrive split across
chunks (``"mail john.do"`` + ``"e@acme.com"``). Each chunk on its own looks
clean, so a per-chunk regex leaks data, and buffering the whole response
destroys time-to-first-token.

``StreamingRedactor`` keeps a small hold-back buffer instead. After each
chunk it finds the last *safe boundary*: a position that no PII match can
span. Everything before that boundary is redacted and released immediately;
only the trailing fragment that could still grow into a match is held.

A position is a safe boundary when the character before it cannot appear
inside any pattern (whitespace, most punctuation). The one exception is card
numbers written in groups (``4111 1111 1111 1111``), whose single spaces
between digits are also bridged. For ordinary prose the held fragment is the
word currently being generated, so the added latency is about one token.

Because no match can cross a safe boundary, redacting the stream piecewise
gives exactly the same output as redacting the complete text at once. The
property-based tests check this equivalence over random chunk splits. The
only exception is a fragment longer than ``max_hold`` with no boundary in
it, which is too long to be a single value and is released in part (see
``StreamingRedactor.feed``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REDACTED = "[REDACTED]"

# Order matters: at a given position the first alternative that matches wins.
# Every repetition is bounded (RFC 5321 local part <= 64, DNS label <= 63) so
# matching stays linear even on long runs of characters with no '@'.
_PII = re.compile(
    r"""
    (?P<email>
        [A-Za-z0-9._%+\-]{1,64} @ [A-Za-z0-9\-]{1,63} (?:\.[A-Za-z0-9\-]{1,63}){0,8} \.[A-Za-z]{2,24}
    )
    | (?P<ssn>
        \b (?!000|666|9\d\d) \d{3} - (?!00) \d{2} - (?!0000) \d{4} \b
    )
    | (?P<card>
        \b (?:\d[ \-]?){12,18} \d \b
    )
    """,
    re.VERBOSE,
)

# Characters that can be part of a match: word characters (any script, so
# regex word boundaries behave identically on a fragment and on the full
# text) plus the punctuation used inside emails, SSNs and card numbers.
_C = r"[\w.%+\-@]"
# The trailing fragment that may still grow into a match: a run of candidate
# characters, optionally preceded by "digits<space>" groups that continue
# with a digit (spaced card numbers such as "4111 1111 1111 1111").
_TRAILING_FRAGMENT = re.compile(rf"(?:{_C}*\d (?=\d))*{_C}*+\Z")

DEFAULT_MAX_HOLD_CHARS = 256  # > RFC 5321 address limit (254), > any SSN/card form


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _replace(match: re.Match[str]) -> str:
    if match.lastgroup == "card":
        digits = re.sub(r"[ \-]", "", match.group())
        if not _luhn_valid(digits):
            return match.group()
    return REDACTED


def redact(text: str) -> str:
    """Redact a complete string. Reference behaviour for the streaming path."""
    return _PII.sub(_replace, text)


def safe_boundary(text: str, max_hold: int = DEFAULT_MAX_HOLD_CHARS) -> int:
    """Index up to which ``text`` can be redacted and released now.

    The search never looks further back than ``max_hold`` characters: a
    longer fragment is cut there by the caller anyway, and the bound keeps
    the cost per chunk constant no matter how much text was released.
    """
    end = len(text)
    if end >= 2 and text[-1] == " " and text[-2].isdigit():
        end -= 1  # the next chunk may continue a spaced digit group
    floor = max(0, end - max_hold - 1)
    match = _TRAILING_FRAGMENT.search(text, floor, end)
    start = match.start() if match else end
    return len(text) if start == end else start


@dataclass
class RedactionStats:
    chars_in: int = 0
    chars_out: int = 0
    max_held: int = 0
    redactions: int = 0


class StreamingRedactor:
    """Incremental redactor. One instance per independent text stream.

    ``feed`` returns the text that is safe to send now (possibly empty);
    ``flush`` must be called at end of stream and returns the remainder.
    Memory is bounded by ``max_hold`` characters regardless of response size.
    """

    def __init__(self, max_hold: int = DEFAULT_MAX_HOLD_CHARS) -> None:
        if max_hold < 32:
            # The longest card format with separators is 23 characters; the
            # default of 256 also covers the longest valid email address.
            raise ValueError("max_hold must be at least 32 characters")
        self._pending = ""
        self._max_hold = max_hold
        self.stats = RedactionStats()

    @property
    def pending(self) -> str:
        return self._pending

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self.stats.chars_in += len(chunk)
        text = self._pending + chunk
        cut = safe_boundary(text, self._max_hold)

        if len(text) - cut > self._max_hold:
            # A fragment this long without a boundary is not a single valid
            # email, SSN or card number. Release all but the last max_hold
            # characters; a match ending inside the kept part is still caught.
            cut = len(text) - self._max_hold

        self._pending = text[cut:]
        self.stats.max_held = max(self.stats.max_held, len(self._pending))
        return self._emit(text[:cut])

    def flush(self) -> str:
        text, self._pending = self._pending, ""
        return self._emit(text)

    def _emit(self, text: str) -> str:
        if not text:
            return ""
        out = redact(text)
        self.stats.chars_out += len(out)
        self.stats.redactions += out.count(REDACTED) - text.count(REDACTED)
        return out
