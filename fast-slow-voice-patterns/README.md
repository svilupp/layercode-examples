# Voice Patterns

Reference implementation of voice agent patterns using fast-slow model orchestration. Demonstrates how to balance responsiveness with accuracy in voice applications using Gemini 2.5 Flash Lite (fast) and Pro (slow) models.

Related blog post: [Building Effective Voice Agents](https://therawaideas.substack.com/p/building-effective-voice-agents)

## Prerequisites

- **Bun** 1.2+ ([install](https://bun.sh))
- **Google AI API Key** with access to Gemini 2.5 Flash Lite and Pro ([get key](https://ai.google.dev))

## Setup

```bash
bun install
cp .env.example .env
```

Edit `.env`:
```bash
GOOGLE_API_KEY=your_google_api_key_here
LAYERCODE_AGENT_ID=your_layercode_agent_id_here
LAYERCODE_WEBHOOK_SECRET=your_layercode_webhook_secret_here
LAYERCODE_API_KEY=your_layercode_api_key_here
```

## Run

```bash
bun run dev
# Server starts on http://localhost:3000
```

Health check: `curl http://localhost:3000/healthz`

## Test

### Quick Test (Single Pattern)
```bash
bun run benchmark:quick /api/agent-echo-relay "What is 2+2?"
```

### Full Benchmark (All Patterns)
```bash
bun run benchmark
```

Results saved to `test/results/benchmark-*.md`

## Benchmark Results

Benchmark latencies (measured 2025-10-19):

| Pattern | Avg TTFB | Avg Total | Use Case |
|---------|----------|-----------|----------|
| **Echo Relay** | 440ms | 11.3s | User feels heard, compliance checks |
| **Shadow** | 356ms | 10.1s | Transparency, show working |
| **Speculative Answerer** | 526ms | 5.0s | Speed + accuracy, FAQ systems |

**TTFB** = Time to First Byte (real LLM streaming, not fake)
**Total** = Complete response time (fast + slow models)

### Why Different Total Times?

- **Echo Relay/Shadow**: Stream fast acknowledgment + wait for full slow response (~10s)
- **Speculative Answerer**: Stream fast answer, slow only verifies (~5s shorter)

## Patterns

### 1. Echo Relay (Parallel)
```
User: "What is 2+2?"
Fast: "You're asking for the sum..." [streams immediately]
Slow: "Two plus two is four." [waits for slow]
```
**Route:** `/api/agent-echo-relay`
**Execution:** Both models start together, fast streams first

### 2. Shadow (Parallel)
```
User: "What is 2+2?"
Fast: "Let me check that..." [streams immediately]
Slow: "Four." [waits for slow]
```
**Route:** `/api/agent-shadow`
**Execution:** Both models start together, fast narrates progress

### 3. Speculative Answerer (Sequential)
```
User: "What's the VAT rate?"
Fast: "Likely 20%..." [streams immediately]
Slow: "Confirmed." OR "Actually, standard rate is 20%, reduced is 5%."
```
**Route:** `/api/agent-speculative-answerer`
**Execution:** Fast completes, THEN slow verifies with structured output

## Key Implementation Details

This is a local only demo -- real applications will pick up a lot of roundtrips and network latency!
We do not use any TTS/STT services in this demo, so the latency is not representative of a real application.

## Project Structure

```
src/
├── lib/ai.ts              # Model config (fastDraft/slowFinal/slowVerify)
├── lib/layercode.ts       # SSE streaming utilities
├── routes/                # Pattern implementations
└── server.ts              # HTTP server

test/
├── lib/                   # TTFB + latency tracking
├── benchmark.ts           # Full comparison
└── quick-test.ts          # Single pattern test
```

## License

MIT
