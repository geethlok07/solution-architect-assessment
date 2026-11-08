#!/usr/bin/env bash
# Starts two mock providers and the router, then walks through each failure mode.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
DEMO_DIR=$(mktemp -d)
export ROUTER_DB_PATH="$DEMO_DIR/llm_router_demo.db"   # fresh database for every run
GATEWAY=http://127.0.0.1:8082/v1/chat/completions

$PY -m task4_model_router.mock_providers --name primary --port 9301 &
P1=$!
$PY -m task4_model_router.mock_providers --name secondary --port 9302 &
P2=$!
$PY -m task4_model_router.app >"$DEMO_DIR/llm_router.log" 2>&1 &
GW=$!
trap 'kill $P1 $P2 $GW 2>/dev/null || true; rm -rf "$DEMO_DIR"' EXIT

for _ in $(seq 1 50); do
  curl -sf http://127.0.0.1:8082/healthz >/dev/null && curl -sf -o /dev/null http://127.0.0.1:9301/docs \
    && curl -sf -o /dev/null http://127.0.0.1:9302/docs && break
  sleep 0.2
done

KEY=$($PY -m task4_model_router.admin create-tenant acme)
SMALL=$($PY -m task4_model_router.admin create-tenant tiny --tokens-per-minute 1500)

mode() { curl -s -X POST "http://127.0.0.1:$1/control" -H 'content-type: application/json' -d "{\"mode\":\"$2\"}" >/dev/null; }
chat() {
  local label=$1 key=$2 max_tokens=${3:-400}
  echo "== $label"
  curl -s -o "$DEMO_DIR/body.json" -D "$DEMO_DIR/headers.txt" -w '   HTTP %{http_code} in %{time_total}s\n' -X POST "$GATEWAY" \
    -H "authorization: Bearer $key" -H 'content-type: application/json' \
    -d "{\"model\":\"gpt-demo\",\"max_tokens\":$max_tokens,\"messages\":[{\"role\":\"user\",\"content\":\"Hello there\"}]}"
  grep -iE '^(x-gateway-route|x-ratelimit-remaining-tokens|retry-after):' "$DEMO_DIR/headers.txt" | sed 's/^/   /' || true
  echo "   $(cat "$DEMO_DIR/body.json")"
}

chat "healthy primary" "$KEY"
chat "tenant 'tiny' (1,500 tokens/min): request reserving ~1,000 tokens" "$SMALL" 1000
chat "tenant 'tiny': second request no longer fits the window -> 429" "$SMALL" 1000
mode 9301 rate_limited; chat "primary returns 429 -> secondary" "$KEY"
mode 9301 slow;         chat "primary hangs for 5s -> secondary after the 3s deadline" "$KEY"
mode 9301 error; mode 9302 error; chat "both providers down -> sanitized 503" "$KEY"
mode 9301 ok; mode 9302 ok
chat "primary recovered, but its circuit is open for 30s -> served by secondary immediately" "$KEY"
chat "invalid API key" "sk-not-a-real-key"
