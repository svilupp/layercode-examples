import { verifyLayercode } from './lib/layercode';
import { handleEchoRelay } from './routes/agent-echo-relay';
import { handleShadow } from './routes/agent-shadow';
import { handleSpeculativeAnswerer } from './routes/agent-speculative-answerer';

type Handler = (request: Request, body: Record<string, unknown>) => Promise<Response>;

const handlers: Record<string, Handler> = {
  '/api/agent-echo-relay': handleEchoRelay,
  '/api/agent-shadow': handleShadow,
  '/api/agent-speculative-answerer': handleSpeculativeAnswerer,
};

const PORT = Number(Bun.env.PORT ?? 3000);

function ok(): Response {
  return new Response('ok', { status: 200 });
}

function notFound(): Response {
  return new Response('Not Found', { status: 404 });
}

function methodNotAllowed(): Response {
  return new Response('Method Not Allowed', { status: 405 });
}

function badRequest(message: string): Response {
  return new Response(message, { status: 400 });
}

export const server = Bun.serve({
  port: PORT,
  idleTimeout: Number(Bun.env.IDLE_TIMEOUT ?? 120),
  fetch: async (request: Request) => {
    const url = new URL(request.url);

    if (request.method === 'GET' && url.pathname === '/healthz') {
      return ok();
    }

    if (request.method !== 'POST') {
      return methodNotAllowed();
    }

    const handler = handlers[url.pathname];
    if (!handler) {
      return notFound();
    }

    let rawBody = '';
    try {
      rawBody = await request.text();
    } catch (error) {
      return badRequest('Unable to read request body');
    }

    let body: Record<string, unknown> = {};
    try {
      body = rawBody.length === 0 ? {} : (JSON.parse(rawBody) as Record<string, unknown>);
    } catch (error) {
      return badRequest('Invalid JSON body');
    }

    try {
      verifyLayercode(request, rawBody);
    } catch (error) {
      return new Response('Unauthorized', { status: 401 });
    }

    return handler(request, body);
  },
});

console.log(`Server listening on http://localhost:${PORT}`);
