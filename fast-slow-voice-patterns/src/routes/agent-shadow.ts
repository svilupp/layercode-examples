import { streamResponse } from '../lib/layercode';
import { fastDraft, slowFinal } from '../lib/ai';

type LayercodeMessage = {
  text?: string;
  [key: string]: unknown;
};

/**
 * Shadow Pattern (Echo Relay Variant)
 * System Controls × Wait for Slow
 *
 * Same as Echo Relay but fast model narrates progress/status
 * instead of restating the request. Provides transparency.
 *
 * TTFB comes from fast model streaming - no fake stream.data() first!
 */
export async function handleShadow(request: Request, body: LayercodeMessage) {
  const userText = (body.text ?? '').toString();

  return streamResponse(body, async ({ stream }) => {
    // Start slow model immediately (system controls - both always run)
    const slowPromise = slowFinal(userText);

    // Fast model narrates progress (THIS creates TTFB)
    const fast = await fastDraft(
      userText,
      'Narrate progress only, such as "Let me check that." Do not provide facts. One sentence.'
    );

    // Stream fast model TTS - this is the first response byte (real TTFB)
    await stream.ttsTextStream(fast.textStream);
    const narrated = (await fast.text).trim();

    // Optional: send metadata after TTS has started
    if (narrated.length > 0) {
      stream.data({ narrated, status: 'working' });
    }

    // Wait for slow model (wait for slow)
    const slow = await slowPromise;
    const slowText = slow.text.trim();
    if (slowText.length > 0) {
      await stream.tts(slowText);
      stream.data({ final: slowText, status: 'complete' });
    }

    stream.end();
  });
}
