import { streamResponse } from '../lib/layercode';
import { fastDraft, slowFinal } from '../lib/ai';

type LayercodeMessage = {
  text?: string;
  [key: string]: unknown;
};

/**
 * Echo Relay Pattern (from blog)
 * System Controls × Wait for Slow
 *
 * Both models always run. Fast provides immediate acknowledgment
 * while slow computes the real answer. Simple and safe.
 */
export async function handleEchoRelay(request: Request, body: LayercodeMessage) {
  const userText = (body.text ?? '').toString();

  return streamResponse(body, async ({ stream }) => {
    // Start slow model immediately (system controls - both always run)
    const slowPromise = slowFinal(userText);

    // Fast model provides immediate acknowledgment
    const fast = await fastDraft(
      userText,
      'Acknowledge and restate the request briefly. No facts. One short sentence.'
    );

    await stream.ttsTextStream(fast.textStream);
    const fastText = (await fast.text).trim();
    stream.data({ provisional: fastText, mode: 'echo-relay' });

    // Wait for slow model (wait for slow)
    const slow = await slowPromise;
    const slowText = slow.text.trim();
    if (slowText.length > 0) {
      await stream.tts(slowText);
      stream.data({ final: slowText });
    }

    stream.end();
  });
}
