# Python Client for LayerCode Evaluation

> NOTE: These scripts have been packaged into production-ready tools:
> - [layercode-gym](https://github.com/svilupp/layercode-gym) - Evaluation framework for LayerCode agents
> - [layercode-create-app](https://github.com/svilupp/layercode-create-app) - CLI to scaffold LayerCode backends with tunneling
>
> Consider using those instead for production use cases.

## Overview

Three Python scripts to programmatically test LayerCode voice agents by simulating browser clients via WebSocket connections. Each script implements a different user input strategy.

**Prerequisites:** Requires a running LayerCode backend server. If you don't have one, use [layercode-create-app](https://github.com/svilupp/layercode-create-app).

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

**Input:** WAV file (default: `data/intro-example-8000hz.wav`)

**Output:** Saves assistant audio to `audio/output_*.wav`

```bash
python simple_file_client.py --agent-id YOUR_AGENT_ID
```

**Key parameters:**
- `--audio-input PATH` - Input WAV file path
- `--chunk-ms 100` - Audio chunk duration (ms)
- `--assistant-idle-timeout 3.0` - Wait time before triggering next turn (s)

---

### 2. simple_tts_client.py
Converts text to speech via OpenAI TTS and streams it.

**Use case:** Scripted conversations with dynamic audio generation.

**Input:** Text string (via `--default-reply` or `$DEFAULT_REPLY`)

**Output:** Saves both user TTS audio (`audio/input_*.wav`) and assistant responses (`audio/output_*.wav`)

```bash
export OPENAI_API_KEY="sk-..."
python simple_tts_client.py \
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

**Input:** AI agent generates contextual replies based on assistant responses

**Output:** Audio artifacts + Logfire traces

```bash
export OPENAI_API_KEY="sk-..."
export LOGFIRE_TOKEN="pylf_..."  # Optional
python simple_ai_client.py \
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

## Setup

### 1. Install dependencies

```bash
cd python-client-for-evaluation
pip install -r requirements.txt
```

**Note:** If `requirements.txt` is missing, install manually:

```bash
pip install httpx websockets loguru python-dotenv openai pydantic-ai logfire
```

### 2. Configure environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
SERVER_URL="http://localhost:8001"      # Your backend server
LAYERCODE_AGENT_ID="agent_xxxxx"        # From LayerCode dashboard
OPENAI_API_KEY="sk-..."                 # For TTS + AI scripts
LOGFIRE_TOKEN="pylf_..."                # Optional: for simple_ai_client.py
```

### 3. Start your backend

Ensure your LayerCode backend is running and accessible at `SERVER_URL`. The client expects a `/api/authorize` endpoint that accepts:

```json
POST /api/authorize
{
  "agent_id": "agent_xxxxx"
}
```

Response:
```json
{
  "client_session_key": "session_key_here",
  "conversation_id": "optional_conversation_id"
}
```

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
