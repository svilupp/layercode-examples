# Python Client for LayerCode Evaluation

> NOTE: These scripts have been packaged into production-ready tools:
> - [layercode-gym](https://github.com/svilupp/layercode-gym) - Evaluation framework for LayerCode agents
> - [layercode-create-app](https://github.com/svilupp/layercode-create-app) - CLI to scaffold LayerCode backends with tunneling
>
> Consider using those instead for production use cases.

## Overview

Three Python scripts to programmatically test LayerCode voice agents by simulating browser clients via WebSocket connections. Each script implements a different user input strategy.

## Quick Start

**1. Configure environment variables:**
```bash
# Set in .env file or export directly
export LAYERCODE_AGENT_ID="agent_xxxxx"     # Required: from LayerCode dashboard
export OPENAI_API_KEY="sk-..."              # Required for TTS + AI scripts
export LOGFIRE_TOKEN="pylf_..."             # Optional for AI script
export SERVER_URL="http://localhost:8001"   # Optional: defaults to localhost:8001
```

**2. Start a LayerCode backend (if you don't have one):**
```bash
uvx layercode-create-app run --tunnel
```

**3. Run any script with uv:**
```bash
# File-based client
uv run simple_file_client.py

# TTS client
uv run simple_tts_client.py --default-reply "Hello, I need help"

# AI client
uv run simple_ai_client.py --max-user-turns 3
```

No need to install dependencies - `uv run` handles everything automatically.

## Architecture

```
+-----------------+
|  Python Client  |
+--------+--------+
         | 1. POST /api/authorize
         |    {agent_id: "..."}
         |
         v
+-----------------+
| Backend Server  |
|  (localhost)    |
+--------+--------+
         | 2. Returns client_session_key
         |
         v
+-----------------+
|  LayerCode WS   |
|  api.layercode  |
|  .com/v1/agents |
+--------+--------+
         | 3. WebSocket connection
         |    ?client_session_key=...
         |
         v
    +------------------------+
    |  Event Loop            |
    |  +------------------+  |
    |  | turn.start       |  |
    |  |   role=user      |  |
    |  +--------+---------+  |
    |           |            |
    |           v            |
    |  +------------------+  |
    |  | client.audio     |  |
    |  | (base64 PCM)     |  |
    |  +--------+---------+  |
    |           |            |
    |           v            |
    |  +------------------+  |
    |  | response.text    |  |
    |  | response.audio   |  |
    |  +------------------+  |
    +------------------------+
```

## Scripts

### 1. simple_file_client.py
Streams a pre-recorded WAV file as user input.

**Use case:** Regression testing with fixed audio samples.

```bash
uv run simple_file_client.py
```

**Key parameters:**
- `--audio-input PATH` - Input WAV file path (default: `data/intro-example-8000hz.wav`)
- `--chunk-ms 100` - Audio chunk duration (ms)
- `--assistant-idle-timeout 3.0` - Wait time before triggering next turn (s)

---

### 2. simple_tts_client.py
Converts text to speech via OpenAI TTS and streams it.

**Use case:** Scripted conversations with dynamic audio generation.

```bash
export OPENAI_API_KEY="sk-..."
uv run simple_tts_client.py \
  --agent-id YOUR_AGENT_ID \
  --default-reply "Hello, I need help with my order"
```

**Key parameters:**
- `--default-reply TEXT` - Text to synthesize
- `--tts-model gpt-4o-mini-tts` - OpenAI TTS model
- `--tts-voice coral` - Voice selection
- `--tts-instructions "Speak slowly"` - Optional voice modulation

---

### 3. simple_ai_client.py
Runs multi-turn conversations with PydanticAI-powered customer persona.

**Use case:** Automated evaluation with AI-generated responses and Logfire observability.

```bash
export OPENAI_API_KEY="sk-..."
export LOGFIRE_TOKEN="pylf_..."  # Optional
uv run simple_ai_client.py \
  --agent-id YOUR_AGENT_ID \
  --max-user-turns 5 \
  --customer-name "Alice"
```

**Key parameters:**
- `--max-user-turns N` - Conversation length (default: 3)
- `--customer-name NAME` - Persona name for AI agent
- `--default-reply TEXT` - Initial greeting (if needed)

**AI Agent behavior:**
- Persona: Polite shopper looking for a blazer
- Context-aware: Reads assistant text responses to generate natural replies
- Instrumented: Full Logfire tracing for PydanticAI + OpenAI calls

---

## Configuration

Environment variables (set in `.env` or pass via CLI flags):

```ini
SERVER_URL="http://localhost:8001"      # Your backend server
LAYERCODE_AGENT_ID="agent_xxxxx"        # From LayerCode dashboard
OPENAI_API_KEY="sk-..."                 # For TTS + AI scripts
LOGFIRE_TOKEN="pylf_..."                # Optional: for simple_ai_client.py
```

Run `uv run <script> --help` for all available options.

---

## Common Options

All scripts support:

- `--server-url URL` - Backend server URL
- `--authorize-path PATH` - Auth endpoint path (default: `/api/authorize`)
- `--ws-url URL` - LayerCode WebSocket URL (default: `wss://api.layercode.com/v1/agents/web/websocket`)
- `--chunk-ms N` - Audio chunk duration in milliseconds
- `--chunk-interval SECS` - Delay between chunks (simulates network conditions)
- `--log-level LEVEL` - Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `--debug` / `--no-debug` - Verbose event logging toggle

---

## Audio Requirements

LayerCode expects:
- **Format:** 16-bit PCM WAV
- **Sample rate:** 8000 Hz (configurable in some scripts)
- **Channels:** Mono
- **Encoding:** base64 over WebSocket

Scripts automatically convert audio to this spec when using TTS.

---

## Troubleshooting

**WebSocket connection fails:**
- Verify `SERVER_URL` is reachable
- Check that `/api/authorize` returns valid `client_session_key`
- Ensure backend has correct `LAYERCODE_AGENT_ID` configured

**TTS scripts fail:**
- Verify `OPENAI_API_KEY` is set and valid
- Check OpenAI API quota/rate limits

**AI client generates poor responses:**
- Adjust persona in `simple_ai_client.py:134-138`
- Modify `--default-reply` for better conversation starters
- Increase `--max-user-turns` for longer conversations

**Audio not saved:**
- Check `audio/` directory permissions
- Verify disk space

---

## Development Notes

These scripts are **intentionally minimal** for evaluation purposes:
- No retry logic on network failures
- Single-conversation lifecycle (no session persistence)
- Synchronous authorization step before WebSocket connection
- Audio buffering uses in-memory bytearrays (not suitable for hours-long sessions)

For production workloads, use [layercode-gym](https://github.com/svilupp/layercode-gym).

---

## License

See parent repository for license information.
