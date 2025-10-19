#!/usr/bin/env bash
set -euo pipefail

SERVER_URL=${SERVER_URL:-http://localhost:3000}

if ! curl --silent --fail --max-time 2 "${SERVER_URL}/healthz" > /dev/null; then
  echo "Server not reachable at ${SERVER_URL}. Start it with 'bun run src/server.ts' in another terminal." >&2
  exit 1
fi

declare -a cases=(
  "/api/agent-echo-relay|Summarize Starlink in one sentence"
  "/api/agent-shadow|Compare iPhone 16 versus 15 camera differences"
  "/api/agent-speculative-answerer|What is the VAT rate in the UK?"
)

for entry in "${cases[@]}"; do
  IFS='|' read -r path prompt <<<"${entry}"
  echo "\n=== Testing ${path} ==="
  bun run test/send-webhook.ts "${path}" "${prompt}"
  echo "=== Completed ${path} ===\n"
  sleep 1
done
