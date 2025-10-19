#!/usr/bin/env bun

import { TestClient } from './lib/test-client';
import { formatDuration } from './lib/metrics';

const [, , pathArg, ...textParts] = process.argv;
const path = pathArg ?? '/api/agent-echo-relay';
const text = textParts.length > 0 ? textParts.join(' ') : 'What are the opening hours tomorrow?';

async function main() {
  const client = new TestClient({ verbose: false });

  console.log('Checking server health...');
  const healthy = await client.healthCheck();
  if (!healthy) {
    console.error('❌ Server not reachable at http://localhost:3000');
    console.error('Start it with: bun run src/server.ts');
    process.exit(1);
  }

  const pattern = path.replace('/api/agent-', '');
  console.log(`\n🧪 Testing pattern: ${pattern}`);
  console.log(`📝 Prompt: "${text}"\n`);

  const result = await client.sendMessage(path, text);

  if (result.error) {
    console.error(`❌ Error: ${result.error}`);
    process.exit(1);
  }

  console.log('📊 Performance Metrics (Client-Side):');
  console.log('─'.repeat(50));
  console.log(`  Time to First Byte:  ${formatDuration(result.timeToFirstByte)}`);
  console.log(`  Total Duration:      ${formatDuration(result.totalDuration)}`);
  console.log('─'.repeat(50));

  console.log('\n✓ Test completed successfully!');
}

main().catch(error => {
  console.error('Test failed:', error);
  process.exit(1);
});
