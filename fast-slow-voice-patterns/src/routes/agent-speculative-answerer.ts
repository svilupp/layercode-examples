import { streamResponse } from '../lib/layercode';
import { fastDraft, slowVerify } from '../lib/ai';

type LayercodeMessage = {
  text?: string;
  [key: string]: unknown;
};

/**
 * Speculative Answerer Pattern (from blog)
 * Fast Decides × Wait for Slow
 *
 * Fast model provides hedged provisional answer immediately.
 * Slow model then verifies/corrects the fast answer.
 *
 * Key difference from Echo/Shadow:
 * - SEQUENTIAL execution (fast completes → then slow verifies)
 * - Slow model receives BOTH user query AND fast answer
 * - Slow returns structured output (needs correction or not)
 *
 * TTFB comes from fast model streaming - real LLM latency!
 */
export async function handleSpeculativeAnswerer(request: Request, body: LayercodeMessage) {
  const userText = (body.text ?? '').toString();

  return streamResponse(body, async ({ stream }) => {
    // Fast model provides provisional answer with hedging (THIS creates TTFB)
    const fast = await fastDraft(
      userText,
      'Provide a quick provisional answer. Hedge clearly ("likely", "probably") and limit to two short sentences.'
    );

    // Stream fast provisional answer - this is the first response byte
    await stream.ttsTextStream(fast.textStream);
    const provisionalAnswer = (await fast.text).trim();

    // Send metadata after TTS has started
    if (provisionalAnswer.length > 0) {
      stream.data({ provisional: provisionalAnswer });
    }

    // NOW invoke slow to verify (sequential, not parallel!)
    // Slow receives BOTH user query and fast answer
    const verification = await slowVerify(userText, provisionalAnswer);

    if (verification.needsCorrection && verification.correctedAnswer) {
      // Fast was wrong - stream correction
      await stream.tts('Actually, ' + verification.correctedAnswer);
      stream.data({
        correction: {
          from: provisionalAnswer,
          to: verification.correctedAnswer,
          reasoning: verification.reasoning
        }
      });
    } else {
      // Fast was correct - confirm it
      await stream.tts('Confirmed.');
      stream.data({ confirmed: true, final: provisionalAnswer });
    }

    stream.end();
  });
}
