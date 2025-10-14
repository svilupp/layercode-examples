"""
Layercode Voice Agent Demo
A simple FastAPI backend that handles voice agent webhooks from Layercode.
"""

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModelSettings

from layercode_sdk import StreamHelper, stream_response, verify_signature

# Load environment variables
load_dotenv()

# Configure Logfire observability
logfire.configure(
    scrubbing=False,
    service_name="python-fastapi-backend",
    send_to_logfire="if-token-present",
    environment="development",
)

# Configuration
TEXT_MODEL = "openai:gpt-5-nano"
SYSTEM_PROMPT = "You are a concise, friendly voice assistant who speaks in short helpful sentences."
WELCOME_MESSAGE = "Hey! How can I help today?"
AUTH_ENDPOINT = "https://api.layercode.com/v1/agents/web/authorize_session"

# Initialize PydanticAI agent
agent = Agent(
    TEXT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    model_settings=OpenAIChatModelSettings(reasoning_effort="minimal", verbosity="low"),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for startup and shutdown."""
    # Startup
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
    logger.info("FastAPI application started")
    logger.info(f"Agent model: {TEXT_MODEL}")

    yield

    # Shutdown
    client: httpx.AsyncClient = app.state.http_client
    await client.aclose()
    logger.info("FastAPI application shutdown")


app = FastAPI(title="Layercode Voice Agent Backend", lifespan=lifespan)

# Instrument FastAPI with Logfire for observability
logfire.instrument_fastapi(app)

# In-memory conversation history (use Redis/DB in production)
_conversation_histories: dict[str, list[ModelMessage]] = {}
_conversation_locks: dict[str, asyncio.Lock] = {}
_lock_registry = asyncio.Lock()


# ============================================================================
# Models
# ============================================================================


class WebhookPayload(BaseModel):
    """Layercode webhook request payload."""

    type: str  # "session.start" | "message" | "session.end" | "session.update"
    session_id: str
    conversation_id: str
    turn_id: str
    text: str | None = None
    metadata: dict[str, Any] | None = None
    from_phone_number: str | None = None
    to_phone_number: str | None = None


class AuthorizeRequest(BaseModel):
    """Request to authorize a client session."""

    agent_id: str
    conversation_id: str | None = None
    metadata: dict[str, Any] | None = None


class AuthorizeResponse(BaseModel):
    """Response with client session key."""

    client_session_key: str
    conversation_id: str
    config: dict[str, Any] | None = None


# ============================================================================
# Conversation History Management
# ============================================================================


async def get_conversation_lock(conversation_id: str) -> asyncio.Lock:
    """Get or create a lock for a conversation to prevent concurrent modifications."""
    async with _lock_registry:
        if conversation_id not in _conversation_locks:
            _conversation_locks[conversation_id] = asyncio.Lock()
        return _conversation_locks[conversation_id]


def append_to_history(conversation_id: str, new_messages: Iterable[ModelMessage]) -> None:
    """Append new messages to conversation history."""
    history = _conversation_histories.get(conversation_id, [])
    _conversation_histories[conversation_id] = list(history) + list(new_messages)


# ============================================================================
# API Routes
# ============================================================================


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/agent", response_model=None)
async def agent_webhook(request: Request) -> JSONResponse | StreamingResponse:
    """
    Main webhook endpoint for Layercode voice agent.
    Handles session.start, message, session.end, and session.update events.
    """
    # Read and verify webhook signature
    body = await request.body()
    signature = request.headers.get("layercode-signature")
    secret = os.getenv("LAYERCODE_WEBHOOK_SECRET")

    if not secret:
        logger.error("LAYERCODE_WEBHOOK_SECRET is not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook secret not configured",
        )

    if not signature or not verify_signature(body.decode("utf-8"), signature, secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    # Parse webhook payload
    try:
        payload = WebhookPayload.model_validate_json(body)
        logger.info(f"Webhook: {payload.type} | conversation: {payload.conversation_id}")
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid webhook payload"
        ) from exc

    # Acquire conversation lock to prevent race conditions
    conversation_lock = await get_conversation_lock(payload.conversation_id)
    await conversation_lock.acquire()

    try:
        # Route to appropriate handler based on event type
        request_dict = payload.model_dump()

        if payload.type == "session.start":
            return await handle_session_start(request_dict)

        elif payload.type == "message":
            return await handle_message(request_dict, payload.conversation_id)

        elif payload.type == "session.end" or payload.type == "session.update":
            return JSONResponse({"status": "ok"})

        else:
            return JSONResponse({"status": "ok"})

    finally:
        conversation_lock.release()


async def handle_session_start(request_body: dict[str, Any]) -> StreamingResponse:
    """Handle session.start event - optionally send a welcome message."""

    async def stream_handler(stream: StreamHelper) -> None:
        # Send metadata if present
        if request_body.get("metadata"):
            stream.data(request_body["metadata"])

        # Send welcome message
        stream.tts(WELCOME_MESSAGE)

        # Optional: Add to conversation history
        # conversation_id = request_body["conversation_id"]
        # welcome_msg = ModelResponse(parts=[TextPart(content=WELCOME_MESSAGE)])
        # append_to_history(conversation_id, [welcome_msg])

        stream.end()

    return await stream_response(request_body, stream_handler)


async def handle_message(request_body: dict[str, Any], conversation_id: str) -> StreamingResponse:
    """Handle message event - generate and stream LLM response."""
    user_text = request_body.get("text")
    if not user_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Missing text in message"
        )

    logger.info(f"Message: {user_text[:100]}")

    # Get conversation history
    history = _conversation_histories.get(conversation_id, [])

    async def stream_handler(stream: StreamHelper) -> None:
        # Send metadata if present
        if request_body.get("metadata"):
            stream.data(request_body["metadata"])

        try:
            # Stream LLM response
            async with agent.run_stream(user_text, message_history=history) as run:
                # Stream text chunks as they're generated
                async for delta in run.stream_text(delta=True):
                    if delta:
                        stream.tts(delta)

                # Get final response and update history
                final_text = await run.get_output()
                logger.info(f"Response: {final_text}")

                # Save conversation history
                append_to_history(conversation_id, run.new_messages())

        except Exception as exc:
            logger.exception("Error generating LLM response")
            error_message = "Sorry, I ran into an issue. Please try again."
            stream.data({"error": str(exc)})
            stream.tts(error_message)

        stream.end()

    return await stream_response(request_body, stream_handler)


@app.post("/api/authorize", response_model=AuthorizeResponse)
async def authorize_session(request_body: AuthorizeRequest) -> JSONResponse:
    """
    Authorize a client session with Layercode.
    This endpoint is called by your frontend to get a session key.
    """
    api_key = os.getenv("LAYERCODE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LAYERCODE_API_KEY is not configured",
        )

    client: httpx.AsyncClient = app.state.http_client

    try:
        response = await client.post(
            AUTH_ENDPOINT,
            json=request_body.model_dump(exclude_none=True),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        logger.info("Session authorized")
        return JSONResponse(status_code=status.HTTP_200_OK, content=response.json())

    except httpx.HTTPStatusError as exc:
        logger.error(f"Layercode API error: {exc.response.status_code}")
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text or str(exc),
        ) from exc

    except httpx.RequestError as exc:
        logger.error(f"Failed to reach Layercode API: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to reach Layercode API",
        ) from exc


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Run the FastAPI application with uvicorn."""
    import uvicorn

    uvicorn.run(
        "api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(int(os.getenv("UVICORN_RELOAD", "0"))),
    )


if __name__ == "__main__":
    main()
