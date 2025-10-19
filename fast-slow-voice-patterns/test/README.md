# Voice Patterns Test Harness

Comprehensive testing infrastructure for comparing LayerCode voice agent patterns with detailed performance metrics.

## Overview

This test harness provides:
- **Performance Benchmarking**: Measure TTFB, TTFT, first TTS, and total duration
- **Token Tracking**: Track tokens consumed by fast and slow models separately
- **Comparison Reports**: Automated markdown reports comparing all patterns
- **LayerCode Compliance**: Adheres to LayerCode webhook SSE API spec

## Architecture

```
test/
├── lib/
│   ├── metrics.ts        # Performance metrics collection
│   ├── test-client.ts    # LayerCode-compliant webhook client
│   └── comparison.ts     # Report generation and comparison
├── benchmark.ts          # Full benchmark orchestrator
├── quick-test.ts         # Single pattern testing
└── results/              # Generated markdown reports
```

## Quick Start

### 1. Start the Server
```bash
bun run dev
```

### 2. Run Quick Test
Test a single pattern:
```bash
bun run benchmark:quick /api/agent-echo "Your question here"
```

### 3. Run Full Benchmark
Test all patterns with multiple test cases:
```bash
bun run benchmark
```

## Metrics Collected

### Timing Metrics
- **TTFB (Time to First Byte)**: Latency until first response byte received
- **TTFT (Time to First Token)**: Time until first TTS or data event
- **First TTS**: Time until first text-to-speech event (when user hears something)
- **Total Duration**: Complete response time from request to stream end

### Resource Metrics
- **Tokens Used**: Tracked separately for fast and slow models
- **Event Count**: Total SSE events, TTS events, and data events

### Timeline
- Detailed event timeline with relative timestamps
- Event types and payloads for debugging

## Understanding Results

### Pattern Comparison

From the latest benchmark (20 tests across 4 patterns):

| Pattern | Avg TTFB | Avg TTFT | Avg Total | Avg Tokens | Best For |
|---------|----------|----------|-----------|------------|----------|
| **Echo** | 371ms | 371ms | 10.64s | 43 | User acknowledgment, feeling heard |
| **Shadow** | 1.4ms | 1.4ms | 11.50s | 36 | Transparency, showing work in progress |
| **Speculative** | 2.4ms | 2.4ms | 9.80s | 60 | Speed, can correct if wrong |
| **Tentative** | 1ms | 1ms | 6.10s | 46 | Reversible transactions, cautious actions |

### Key Insights

1. **Fastest Response Time**: Tentative (6.10s avg)
   - Commits or rolls back quickly
   - Good for action-oriented queries

2. **Most Token Efficient**: Shadow (36 tokens avg)
   - Simple progress narration
   - Lets slow model do the heavy lifting

3. **Fastest First Byte**: Tentative/Shadow (~1ms)
   - Data events sent immediately
   - User knows something is happening

4. **Most Token Usage**: Speculative (60 tokens avg)
   - Both fast and slow models generate full responses
   - Higher cost but better correction capability

## Customizing Tests

### Add Custom Test Cases

Edit `test/benchmark.ts`:
```typescript
const TEST_CASES = [
  'Your custom prompt 1',
  'Your custom prompt 2',
  // ...
];
```

### Test Specific Patterns

```bash
# Test only echo pattern
bun run benchmark:quick /api/agent-echo "Test prompt"

# Test only shadow pattern
bun run benchmark:quick /api/agent-shadow "Test prompt"
```

### Adjust Benchmark Settings

In `test/benchmark.ts`:
```typescript
const client = new TestClient({
  baseUrl: 'http://localhost:3000',
  timeout: 60000,    // Request timeout (ms)
  verbose: true,     // Enable detailed logging
});
```

## Report Format

Generated reports include:

1. **Summary Statistics**: Averages across all test cases per pattern
2. **Best Performers**: Fastest pattern for each metric
3. **Detailed Results**: Per-test-case breakdown with event timelines
4. **Metrics Glossary**: Explanation of all metrics

Example report: `test/results/benchmark-2025-10-19T11-15-28.md`

## Integration with LayerCode

The test harness sends LayerCode-compliant webhook payloads:

```typescript
{
  "type": "message",
  "session_id": "sess_test",
  "conversation_id": "conv_test",
  "turn_id": "turn_1234567890",
  "text": "Your prompt here"
}
```

All patterns return SSE streams with:
- `response.tts`: Text-to-speech events
- `response.data`: Metadata and structured data
- `response.end`: End of stream

## Development

### Adding New Metrics

1. Update `PatternMetrics` type in `test/lib/metrics.ts`
2. Implement collection in `recordEvent()`
3. Aggregate in `aggregateMetrics()`
4. Display in `generateMarkdownReport()`

### Testing New Patterns

1. Add route handler to `src/routes/`
2. Register in `src/server.ts`
3. Add to `PATTERNS` array in `test/benchmark.ts`

## Troubleshooting

### Server Not Reachable
```
❌ Server not reachable at http://localhost:3000
```
**Solution**: Start the server with `bun run dev`

### Token Counts Missing
**Issue**: Token usage shows as "N/A"

**Solution**: Ensure patterns call `resetTokenTracking()` and `getTokenUsage()` from `src/lib/ai.ts`

### Slow Benchmarks
**Issue**: Benchmark takes too long

**Solution**: Reduce test cases in `test/benchmark.ts` or increase timeout

## Best Practices

1. **Run benchmarks on consistent hardware** for reproducible results
2. **Clear caches** before running benchmarks
3. **Multiple runs** to account for variance
4. **Document changes** to test cases in git commits
5. **Compare reports** over time to track regressions

## Future Enhancements

Potential improvements:
- [ ] Statistical analysis (std dev, percentiles)
- [ ] Cost estimation based on token pricing
- [ ] Latency distribution histograms
- [ ] A/B testing framework
- [ ] Load testing capabilities
- [ ] Real LayerCode integration tests
