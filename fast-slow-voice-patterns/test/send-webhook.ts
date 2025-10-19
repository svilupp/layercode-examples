const [, , pathArg = '/api/agent-echo', ...textParts] = process.argv;
const text = textParts.length > 0 ? textParts.join(' ') : 'What are the opening hours tomorrow?';

const payload = {
  type: 'message',
  session_id: 'sess_local',
  conversation_id: 'conv_local',
  turn_id: `turn_${Date.now()}`,
  text,
};

const response = await fetch(`http://localhost:3000${pathArg}`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(payload),
});

if (!response.ok || !response.body) {
  console.error('HTTP error', response.status, await response.text());
  process.exit(1);
}

console.log('--- SSE stream start ---');
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
  const { value, done } = await reader.read();
  if (done) {
    break;
  }

  buffer += decoder.decode(value, { stream: true });

  let doubleNewlineIndex = buffer.indexOf('\n\n');
  while (doubleNewlineIndex !== -1) {
    const chunk = buffer.slice(0, doubleNewlineIndex);
    buffer = buffer.slice(doubleNewlineIndex + 2);

    chunk
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .forEach((line) => {
        if (line.startsWith('data:')) {
          const data = line.slice(5).trim();
          try {
            const parsed = JSON.parse(data);
            console.log('event', parsed.type ?? 'data', parsed);
          } catch (error) {
            console.log('data', data);
          }
        }
      });

    doubleNewlineIndex = buffer.indexOf('\n\n');
  }
}

console.log('--- SSE stream end ---');
