#!/usr/bin/env bun

import { TestClient } from './lib/test-client';
import { compareResults, printComparisonTable, generateMarkdownReport } from './lib/comparison';
import type { AggregatedMetrics } from './lib/metrics';

const PATTERNS = [
  '/api/agent-echo-relay',
  '/api/agent-shadow',
  '/api/agent-speculative-answerer',
];

const TEST_CASES = [
  'Summarize Starlink in one sentence',
  'What is the VAT rate in the UK?',
  'Book a table for 2 at 7pm tonight',
];

async function main() {
  console.log('🚀 Voice Pattern Performance Benchmark\n');
  console.log('Metrics: TTFB (Time to First Byte) + Total Latency (client-side, real LLM latency)\n');

  const client = new TestClient({
    baseUrl: process.env.SERVER_URL ?? 'http://localhost:3000',
    verbose: true,
  });

  console.log('Checking server health...');
  const healthy = await client.healthCheck();
  if (!healthy) {
    console.error('❌ Server not reachable. Start it with: bun run src/server.ts');
    process.exit(1);
  }
  console.log('✓ Server is healthy\n');

  const results: AggregatedMetrics[] = [];
  const totalTests = PATTERNS.length * TEST_CASES.length;
  let completed = 0;

  console.log(`Running ${totalTests} tests (${PATTERNS.length} patterns × ${TEST_CASES.length} test cases)...\n`);

  for (const testCase of TEST_CASES) {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`Test Case: "${testCase}"`);
    console.log('='.repeat(80));

    for (const pattern of PATTERNS) {
      const patternName = pattern.replace('/api/agent-', '');
      console.log(`\n[${++completed}/${totalTests}] Testing ${patternName}...`);

      try {
        const result = await client.sendMessage(pattern, testCase);
        results.push(result);

        const status = result.error ? '❌' : '✓';
        console.log(`${status} ${patternName}`);
      } catch (error) {
        console.error(`❌ ${patternName} failed:`, error);
        results.push({
          pattern: patternName,
          prompt: testCase,
          error: error instanceof Error ? error.message : String(error),
        });
      }

      // Small delay between tests
      await Bun.sleep(500);
    }
  }

  console.log('\n' + '='.repeat(80));
  console.log('All tests completed!');
  console.log('='.repeat(80) + '\n');

  // Generate comparison report
  const report = compareResults(results);
  printComparisonTable(report);

  // Save to file
  const markdownReport = generateMarkdownReport(report);
  const timestamp = new Date().toISOString().replace(/:/g, '-').split('.')[0];
  const reportPath = `test/results/benchmark-${timestamp}.md`;

  try {
    await Bun.write(reportPath, markdownReport);
    console.log(`\n📊 Report saved to: ${reportPath}`);
  } catch (error) {
    console.log('\n📊 Report generated (file save failed, results printed above)');
  }

  // Exit with error code if any tests failed
  const errorCount = results.filter(r => r.error).length;
  if (errorCount > 0) {
    console.log(`\n⚠️  ${errorCount}/${totalTests} tests failed`);
    process.exit(1);
  }

  console.log(`\n✓ All ${totalTests} tests passed!`);
}

main().catch(error => {
  console.error('Benchmark failed:', error);
  process.exit(1);
});
