# Task 3 — LLM Gateway Streaming Guardrail (PII redaction)

An OpenAI-compatible `POST /v1/chat/completions` proxy that streams the provider's response back to the client while redacting emails, US Social Security numbers and payment card numbers in real time, replacing each with `[REDACTED]`. The full response is never accumulated: memory per stream is constant and the guardrail adds about 4 ms to time-to-first-token.

```
client ◀── SSE ── gateway ◀── SSE ── LLM provider
                    │
   bytes ─▶ SSEParser (incremental UTF-8) ─▶ JSON chunk ─▶ StreamingRedactor per choice
                                                           (delta.content only)
                                                         ─▶ re-encoded SSE event ─▶ client
```

## Run

Start each server in its own terminal, from the repository root:

```bash
pip install -e ".[dev]"
python -m task3_llm_guardrail.mock_llm       # terminal 1: mock provider on :9200
python -m task3_llm_guardrail.gateway        # terminal 2: guardrail on :8081

# terminal 3
curl -N localhost:8081/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mock-assistant","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

Reconstructed output of that stream:

```
Thanks for reaching out! I found the account. The primary contact is [REDACTED] and the SSN on
file is [REDACTED]. The card ending in 1111 (full number [REDACTED]) was charged twice, so I have
issued a refund. You can also reach billing at [REDACTED]. Order 4111-2222 is unaffected.
```

The `mock-echo` model replies with the user's message, which makes it easy to try arbitrary input. To put a real provider behind the gateway, set `LLM_UPSTREAM_URL` (any OpenAI-compatible endpoint) and `LLM_UPSTREAM_API_KEY`.

## The core problem: PII split across chunks

Providers stream a few characters at a time, so a sensitive value usually arrives in pieces: `"mail john.do"` then `"e@acme.com"`. Each piece looks clean on its own, so running a regex per chunk leaks data. Buffering the whole response fixes that but destroys time-to-first-token.

### Hold-back at the last safe boundary

[`redactor.py`](redactor.py) keeps a small pending buffer. After each chunk it finds the **last safe boundary**: a position no PII match can span. Text before that boundary is redacted and released at once; only the trailing fragment that could still grow into a match is held.

- A character that can never be part of a match (whitespace, most punctuation) makes a safe boundary.
- Card numbers written in groups (`4111 1111 1111 1111`) are the one pattern containing spaces, so a space between digits is bridged, and a trailing `digit + space` is held in case the next chunk continues the group.
- The boundary search only examines `pending + chunk` and never looks back further than the hold limit, so the cost per chunk is constant regardless of how much text has already gone out.

```
feed("Hello")        -> ""              (could still become Hello@x.com)
feed(" world")       -> "Hello "
feed(", how")        -> "world, "
feed(" mail john.do")-> "how mail "
feed("e@acme.com ")  -> "[REDACTED] "
```

For ordinary prose the held fragment is the word currently being generated, so the added latency is roughly one token.

**Equivalence guarantee:** because no match can cross a safe boundary, redacting piecewise yields exactly the same output as redacting the complete text in one go. Hypothesis property tests check this over hundreds of random texts with random chunk splits, plus every possible two-way split of a sample. The one deliberate exception is the hold limit described next.

**Bounded memory:** a fragment longer than the hold limit (256 characters, above the RFC 5321 maximum address length of 254) cannot be a single email, SSN or card number, so its oldest part is released. Peak memory while streaming 5 MB is about 6 KiB.

### Patterns

| Type | Rule | False-positive guard |
| --- | --- | --- |
| Email | `local@domain.tld`, local part ≤ 64, labels ≤ 63 | Requires a dotted domain with an alphabetic TLD |
| US SSN | `ddd-dd-dddd` | Rejects area `000`, `666`, `9xx`, group `00`, serial `0000`; ISO dates do not match |
| Payment card | 13–19 digits, optional single spaces or dashes | Luhn checksum must pass, so order numbers and phone numbers survive |

All repetitions are bounded and the boundary scan uses a possessive quantifier, so the regex work is linear even on hostile input (a 400 KB word, endless `1 1 1 1`, repeated `a@`). There is a test for each of these.

## Gateway mechanics

- **Byte-level SSE parsing** ([`sse.py`](sse.py)) with an incremental UTF-8 decoder: a multi-byte character split across network reads is reassembled, never corrupted. CR, LF and CRLF (including a CRLF split across reads) are all handled.
- **Only `choices[].delta.content` is rewritten.** Role chunks, `finish_reason`, usage, keep-alive comments and `[DONE]` pass through. Each choice index (`n > 1`) gets its own redactor.
- **Flushing:** held text for a choice is released in the chunk that carries its `finish_reason`; anything left when `[DONE]` arrives or the upstream closes is emitted as a final content chunk before `[DONE]`.
- **No empty frames:** a chunk whose content is entirely held back is not sent.
- **Other events:** JSON events that are not chat chunks (an error object, for example) are scrubbed with the batch redactor before being forwarded. Events whose data is not JSON are dropped and logged rather than passed through unchecked.
- **Malformed chunks** (a choice that is not an object, a non-numeric index, a missing delta) are tolerated without breaking the stream.
- **No buffering anywhere:** async generator into `StreamingResponse`, `Cache-Control: no-cache` and `X-Accel-Buffering: no` so intermediaries do not buffer either.
- **Cancellation:** if the client disconnects, the generator is cancelled and the upstream connection is closed in `finally`, so the provider stops generating tokens nobody will read.
- **Errors are sanitized:** upstream 4xx statuses that are meaningful to the client (400, 404, 413, 422, 429) are passed on with a generic message; other failures become 502, timeouts 504. Provider bodies, hostnames and stack traces never reach the client.
- **Non-streaming requests** (`stream: false`) are redacted in full before being returned.
- Each stream logs time-to-first-byte, number of redactions and the largest hold-back.

## Performance

`python -m task3_llm_guardrail.bench` on a single core:

| Measurement | Result |
| --- | --- |
| Redactor throughput (4-character chunks) | ≈ 2 MB/s ≈ 500,000 chunks/s |
| Cost per chunk, p50 / p99 | ≈ 1.7 µs / ≈ 7 µs |
| Peak memory streaming 5 MB | ≈ 6 KiB |
| Largest hold-back on realistic text | 37 characters |
| Added time-to-first-token (localhost, p50) | ≈ 4 ms |
| Added total stream time (localhost, p50) | ≈ 4 ms |

A model streams on the order of 100 tokens per second, so one core of redaction work covers thousands of concurrent streams; the gateway is I/O bound.

## Tests

```bash
pytest task3_llm_guardrail
```

- `test_redactor.py` — every pattern and its false-positive guards; every two-way split and single-character streaming; property tests (streamed == batch, injected PII never survives any chunking); bounded memory; adversarial input in linear time.
- `test_sse.py` — random byte-level splits, CRLF across reads, multi-byte characters across reads, comments, multi-line data.
- `test_gateway.py` — end-to-end redaction for 25 different upstream chunkings, stream framing preserved, non-streaming path, multi-choice streams, flush without `finish_reason`, malformed chunks, error sanitization, and a real-socket test proving the first words reach the client within 300 ms of a 2-second stream (no buffering).

## Limitations and next steps

- Regexes cover structured identifiers. Names, addresses and free-form PII need an NER model (for example Microsoft Presidio); it would run as a second, slower pass on the same safe-boundary segments.
- Tool-call arguments (`delta.tool_calls[].function.arguments`) are passed through; the same redactor can be applied per tool-call index.
- Patterns are US-centric; national ID formats and IBANs would be added as further alternatives in the same compiled pattern.
- Redaction is one-way. A reversible tokenization vault would let trusted downstream systems recover values.
