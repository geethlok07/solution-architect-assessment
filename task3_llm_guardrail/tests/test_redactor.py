from __future__ import annotations

import re
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from task3_llm_guardrail.redactor import REDACTED, StreamingRedactor, redact, safe_boundary


def stream(text: str, sizes: list[int], max_hold: int = 256) -> str:
    redactor = StreamingRedactor(max_hold=max_hold)
    out, index = [], 0
    for size in sizes:
        out.append(redactor.feed(text[index : index + size]))
        index += size
    out.append(redactor.feed(text[index:]))
    out.append(redactor.flush())
    return "".join(out)


# --- pattern coverage --------------------------------------------------------


@pytest.mark.parametrize(
    "email",
    [
        "jane.doe@example.com",
        "a@b.co",
        "billing-team+refunds@acme-corp.io",
        "first_last%tag@sub.domain.example.org",
        "UPPER.Case@Example.COM",
    ],
)
def test_emails_are_redacted(email: str) -> None:
    assert redact(f"write to {email} today") == f"write to {REDACTED} today"


@pytest.mark.parametrize("ssn", ["123-45-6789", "078-05-1120", "899-99-9999"])
def test_valid_ssns_are_redacted(ssn: str) -> None:
    assert redact(f"SSN: {ssn}.") == f"SSN: {REDACTED}."


@pytest.mark.parametrize(
    "not_ssn",
    [
        "000-12-3456",  # area 000 is never issued
        "666-12-3456",  # area 666 is never issued
        "912-12-3456",  # 9xx is reserved
        "123-00-4567",  # group 00
        "123-45-0000",  # serial 0000
        "1234-56-7890",  # wrong shape
        "2024-11-06",  # ISO date
    ],
)
def test_invalid_ssn_shapes_are_left_alone(not_ssn: str) -> None:
    assert redact(f"ref {not_ssn}") == f"ref {not_ssn}"


@pytest.mark.parametrize(
    "card",
    [
        "4111111111111111",  # Visa test number
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "5500 0000 0000 0004",  # Mastercard
        "3782 822463 10005",  # Amex, 15 digits
        "6011000990139424",  # Discover
    ],
)
def test_luhn_valid_cards_are_redacted(card: str) -> None:
    assert redact(f"card {card} on file") == f"card {REDACTED} on file"


@pytest.mark.parametrize(
    "number",
    [
        "4111111111111112",  # fails Luhn
        "1234 5678 9012 3456",  # fails Luhn
        "4111-2222",  # too short
        "+1 (415) 555-0100",  # phone number
        "order 1000000",
        "v1.2.3",
    ],
)
def test_numbers_that_are_not_cards_are_left_alone(number: str) -> None:
    assert redact(number) == number


def test_multiple_values_in_one_text() -> None:
    text = "Email a@b.co, SSN 123-45-6789 and card 4111 1111 1111 1111."
    assert redact(text) == f"Email {REDACTED}, SSN {REDACTED} and card {REDACTED}."


# --- streaming behaviour -----------------------------------------------------

SAMPLE = (
    "Hi! Contact jane.doe@example.com, SSN 123-45-6789, card 4111 1111 1111 1111 "
    "or 5500-0000-0000-0004. Not a card: 4111 1111 1111 1112. Done: a@b.co"
)


def test_every_two_way_split_matches_batch_redaction() -> None:
    expected = redact(SAMPLE)
    for split in range(len(SAMPLE) + 1):
        assert stream(SAMPLE, [split]) == expected, f"split at {split}"


def test_single_character_stream_matches_batch_redaction() -> None:
    assert stream(SAMPLE, [1] * len(SAMPLE)) == redact(SAMPLE)


def test_email_split_across_chunks_never_leaks() -> None:
    redactor = StreamingRedactor()
    released = redactor.feed("mail john.do") + redactor.feed("e@acme.com") + redactor.feed(" now")
    released += redactor.flush()
    assert released == f"mail {REDACTED} now"
    assert "john" not in released


def test_spaced_card_is_held_across_group_boundaries() -> None:
    redactor = StreamingRedactor()
    released = ""
    for piece in ["card 4111 ", "1111 ", "1111 ", "1111", " thanks"]:
        released += redactor.feed(piece)
        assert "4111" not in released
    assert released + redactor.flush() == f"card {REDACTED} thanks"


def test_prose_is_released_with_at_most_one_word_of_delay() -> None:
    redactor = StreamingRedactor()
    assert redactor.feed("Hello") == ""  # could still become "Hello@x.com"
    assert redactor.feed(" world") == "Hello "
    assert redactor.feed(", how") == "world, "
    assert redactor.feed(" are you?") == "how are you?"
    assert redactor.pending == ""


def test_text_without_candidates_is_released_immediately() -> None:
    redactor = StreamingRedactor()
    assert redactor.feed("!!! ??? ") == "!!! ??? "


def test_memory_is_bounded_for_unbroken_input() -> None:
    redactor = StreamingRedactor(max_hold=64)
    released = "".join(redactor.feed("x" * 1000) for _ in range(1000))
    assert len(redactor.pending) == 64
    assert redactor.stats.max_held <= 64
    assert len(released) + len(redactor.flush()) == 1_000_000


@pytest.mark.parametrize(
    "hostile",
    [
        pytest.param("a" * 400_000 + " tail", id="one-huge-word"),
        pytest.param(("x" * 200 + ".") * 2_000, id="long-dotted-runs"),
        pytest.param("1 " * 200_000, id="endless-spaced-digits"),
        pytest.param("a@" * 200_000 + "b", id="repeated-at-signs"),
        pytest.param("4111-" * 80_000, id="endless-dashed-digits"),
    ],
)
def test_adversarial_input_is_processed_in_linear_time(hostile: str) -> None:
    started = time.perf_counter()
    for size in (1, 7, 4096, len(hostile)):
        redactor = StreamingRedactor()
        limit = len(hostile) if size >= 4096 else 100_000
        for index in range(0, limit, size):
            redactor.feed(hostile[index : index + size])
        redactor.flush()
        assert redactor.pending == ""
    assert time.perf_counter() - started < 10


def test_max_hold_must_cover_the_longest_pattern() -> None:
    with pytest.raises(ValueError):
        StreamingRedactor(max_hold=8)


def test_safe_boundary_examples() -> None:
    assert safe_boundary("hello world") == 6
    assert safe_boundary("hello world ") == 12
    assert safe_boundary("card 4111 ") == 5
    assert safe_boundary("card 4111 1111") == 5
    assert safe_boundary("in 2024 we") == 8


def test_stats_count_redactions() -> None:
    redactor = StreamingRedactor()
    redactor.feed(SAMPLE)
    redactor.flush()
    assert redactor.stats.redactions == 5


# --- property-based equivalence ----------------------------------------------

PII_SAMPLES = [
    "jane.doe@example.com",
    "x_y+z@mail.co.uk",
    "123-45-6789",
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "5500000000000004",
]
NOISE = st.sampled_from(
    [" ", "  ", ".", ",", "-", "@", "(", ")", "\n", "é", "9", "12", "abc", "Order", " 4111", "1111 ", ": "]
)
fragments = st.lists(st.one_of(st.sampled_from(PII_SAMPLES), NOISE, st.text(max_size=6)), max_size=40)


@settings(max_examples=400, deadline=None)
@given(fragments, st.lists(st.integers(min_value=0, max_value=12), max_size=60))
def test_streamed_output_equals_batch_output(parts: list[str], sizes: list[int]) -> None:
    # A hold limit above the generated text length isolates the boundary logic
    # from the forced cut, which only applies to fragments no PII can fill.
    text = "".join(parts)
    assert stream(text, sizes, max_hold=4096) == redact(text)


@settings(max_examples=300, deadline=None)
@given(
    st.lists(st.sampled_from(["Sure, ", "the value is ", "ok. ", "contact: ", "\n"]), max_size=6),
    st.sampled_from(PII_SAMPLES),
    st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=40),
)
def test_injected_pii_never_survives_any_chunking(prefix: list[str], secret: str, sizes: list[int]) -> None:
    text = "".join(prefix) + secret + " and more text."
    output = stream(text, sizes)
    assert secret not in output
    digits = re.sub(r"\D", "", secret)
    if len(digits) >= 9:
        assert digits not in re.sub(r"\D", "", output)
