#!/usr/bin/env bash
# Starts the mock MCP server and the gateway, then exercises the main paths with curl.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
GATEWAY=http://127.0.0.1:8080/mcp
DEMO_DIR=$(mktemp -d)

$PY -m task2_mcp_gateway.mock_mcp_server >"$DEMO_DIR/mock_mcp.log" 2>&1 &
MOCK_PID=$!
$PY -m task2_mcp_gateway.gateway >"$DEMO_DIR/mcp_gateway.log" 2>&1 &
GATEWAY_PID=$!
trap 'kill $MOCK_PID $GATEWAY_PID 2>/dev/null || true; rm -rf "$DEMO_DIR"' EXIT

for _ in $(seq 1 50); do
  curl -sf http://127.0.0.1:8080/healthz >/dev/null && curl -sf -o /dev/null -X POST http://127.0.0.1:9100/mcp \
    -H 'content-type: application/json' -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' && break
  sleep 0.2
done

ADMIN=$($PY -m task2_mcp_gateway.auth --role admin --subject alice)
VIEWER=$($PY -m task2_mcp_gateway.auth --role viewer --subject bob)

call() {
  local label=$1 token=$2 body=$3
  echo "== $label"
  curl -s -w '\n   HTTP %{http_code}\n' -X POST "$GATEWAY" \
    -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' \
    ${token:+-H "authorization: Bearer $token"} -d "$body"
}

call "no token"                       ""        '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
call "viewer: tools/list (forwarded)" "$VIEWER" '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
call "viewer: get_customer_record"    "$VIEWER" '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_customer_record","arguments":{"customer_id":"CUST-10001"}}}'
call "viewer: admin_reset_key (denied)" "$VIEWER" '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"tenant_id":"t-1"}}}'
call "admin: admin_reset_key"         "$ADMIN"  '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"tenant_id":"t-1"}}}'
call "malformed JSON"                 "$ADMIN"  '{"jsonrpc":"2.0","id":6,'

echo
echo "== audit log"
grep audit "$DEMO_DIR/mcp_gateway.log" || true
