"""
Layercode Voice Agent - Self-Contained FastHTML App
A simple chatbot with microphone support using FastHTML and HTMX.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
import logfire
from dotenv import load_dotenv
import fasthtml.common
from fasthtml.common import Script, fast_app, Title, P, Form, Input
from monsterui.all import (
    Theme,
    DivLAligned,
    DivCentered,
    Label,
    LabelT,
    Button,
    ButtonT,
    Card,
    CardT,
    CardContainer,
    CardBody,
    Alert,
    AlertT,
    Container,
    ContainerT,
    TextPresets,
    UkIcon,
)
from loguru import logger
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModelSettings
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from layercode_sdk import StreamHelper, stream_response, verify_signature

load_dotenv()

logfire.configure(
    scrubbing=False,
    service_name="fasthtml-layercode-app",
    send_to_logfire="if-token-present",
    environment="development",
)
logfire.instrument_pydantic_ai()

BASE_DIR = Path(__file__).resolve().parent
TEXT_MODEL = "openai:gpt-5-nano"
SYSTEM_PROMPT = (
    "You are a concise, friendly voice assistant who speaks in short helpful sentences."
)
WELCOME_MESSAGE = "Hey! How can I help today?"
AUTH_ENDPOINT = "https://api.layercode.com/v1/agents/web/authorize_session"

agent = Agent(
    TEXT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    model_settings=OpenAIChatModelSettings(reasoning_effort="minimal", verbosity="low"),
)

_conversation_histories: dict[str, list[ModelMessage]] = {}
_conversation_locks: dict[str, asyncio.Lock] = {}
_lock_registry = asyncio.Lock()


class WebhookPayload(BaseModel):
    """Layercode webhook request payload."""

    type: str
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
    sdk_version: str | None = None


class AuthorizeResponse(BaseModel):
    """Response with client session key."""

    client_session_key: str
    conversation_id: str
    config: dict[str, Any] | None = None


async def get_conversation_lock(conversation_id: str) -> asyncio.Lock:
    """Get or create a lock for a conversation to prevent concurrent modifications."""
    async with _lock_registry:
        if conversation_id not in _conversation_locks:
            _conversation_locks[conversation_id] = asyncio.Lock()
        return _conversation_locks[conversation_id]


def append_to_history(conversation_id: str, new_messages: list[ModelMessage]) -> None:
    """Append new messages to conversation history."""
    history = _conversation_histories.get(conversation_id, [])
    _conversation_histories[conversation_id] = list(history) + list(new_messages)


async def lifespan_context(app):
    """Lifespan context manager for startup and shutdown."""
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
    logger.info("FastHTML application started")
    logger.info(f"Agent model: {TEXT_MODEL}")

    yield

    client: httpx.AsyncClient = app.state.http_client
    await client.aclose()
    logger.info("FastHTML application shutdown")


LAYERCODE_SDK_SCRIPT = Script(
    src="https://cdn.jsdelivr.net/npm/@layercode/js-sdk@2.3.0/dist/layercode-js-sdk.min.js"
)

app, rt = fast_app(
    hdrs=[
        *Theme.blue.headers(),
        LAYERCODE_SDK_SCRIPT,
    ],
    lifespan=lifespan_context,
)

logfire.instrument_fastapi(app)


@rt("/health")
def health():
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


@rt("/api/authorize")
async def authorize(request: Request):
    """Authorize a client session with Layercode."""
    api_key = os.getenv("LAYERCODE_API_KEY")
    if not api_key:
        return JSONResponse(
            {"detail": "LAYERCODE_API_KEY is not configured"},
            status_code=500,
        )

    try:
        body = await request.json()
        auth_request = AuthorizeRequest.model_validate(body)
    except ValidationError as exc:
        return JSONResponse(
            {"detail": "Invalid request body", "errors": exc.errors()},
            status_code=400,
        )

    client: httpx.AsyncClient = app.state.http_client

    layercode_request = {
        "pipeline_id": auth_request.agent_id,
    }
    if auth_request.conversation_id:
        layercode_request["conversation_id"] = auth_request.conversation_id
    if auth_request.metadata:
        layercode_request["metadata"] = auth_request.metadata

    try:
        response = await client.post(
            AUTH_ENDPOINT,
            json=layercode_request,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        logger.info("Session authorized")
        return JSONResponse(response.json(), status_code=200)

    except httpx.HTTPStatusError as exc:
        logger.error(f"Layercode API error: {exc.response.status_code}")
        return JSONResponse(
            {"detail": exc.response.text or str(exc)},
            status_code=exc.response.status_code,
        )

    except httpx.RequestError as exc:
        logger.error(f"Failed to reach Layercode API: {exc}")
        return JSONResponse(
            {"detail": "Unable to reach Layercode API"},
            status_code=502,
        )


@rt("/api/agent")
async def agent_webhook(request: Request):
    """Main webhook endpoint for Layercode voice agent."""
    body = await request.body()
    signature = request.headers.get("layercode-signature")
    secret = os.getenv("LAYERCODE_WEBHOOK_SECRET")

    if not secret:
        logger.error("LAYERCODE_WEBHOOK_SECRET is not configured")
        return JSONResponse(
            {"detail": "Webhook secret not configured"},
            status_code=500,
        )

    if not signature or not verify_signature(body.decode("utf-8"), signature, secret):
        return JSONResponse({"detail": "Invalid signature"}, status_code=401)

    try:
        payload = WebhookPayload.model_validate_json(body)
        logger.info(
            f"Webhook: {payload.type} | conversation: {payload.conversation_id}"
        )
    except ValidationError:
        return JSONResponse(
            {"detail": "Invalid webhook payload"},
            status_code=400,
        )

    conversation_lock = await get_conversation_lock(payload.conversation_id)

    async with conversation_lock:
        request_dict = payload.model_dump()

        if payload.type == "session.start":
            return await handle_session_start(request_dict)

        elif payload.type == "message":
            return await handle_message(request_dict, payload.conversation_id)

        elif payload.type == "session.end" or payload.type == "session.update":
            return JSONResponse({"status": "ok"})

        else:
            return JSONResponse({"status": "ok"})


async def handle_session_start(request_body: dict[str, Any]) -> StreamingResponse:
    """Handle session.start event - optionally send a welcome message."""

    async def stream_handler(stream: StreamHelper) -> None:
        # Send metadata if present
        if request_body.get("metadata"):
            stream.data(request_body["metadata"])

        # Send welcome message
        stream.tts(WELCOME_MESSAGE)

        stream.end()

    return await stream_response(request_body, stream_handler)


async def handle_message(
    request_body: dict[str, Any], conversation_id: str
) -> StreamingResponse:
    """Handle message event - generate and stream LLM response."""
    user_text = request_body.get("text")
    if not user_text:
        return JSONResponse(
            {"detail": "Missing text in message"},
            status_code=400,
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


# ============================================================================
# Chat UI
# ============================================================================


@rt("/")
def index():
    """Main chat interface."""
    # Load agent ID from environment variable
    agent_id = os.getenv("LAYERCODE_AGENT_ID", "").strip()

    if not agent_id:
        return (
            Title("Layercode Voice Chat"),
            Container(
                DivCentered(
                    Card(
                        Alert(
                            "Configure LAYERCODE_AGENT_ID in .env to use the chat UI.",
                            cls=AlertT.warning,
                        ),
                        cls=CardT.default,
                    ),
                    cls="min-h-screen",
                ),
                cls=ContainerT.lg,
            ),
        )

    # Load metadata if present
    metadata = {}
    metadata_raw = os.getenv("LAYERCODE_FRONTEND_METADATA")
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in LAYERCODE_FRONTEND_METADATA")

    config_payload = {
        "agentId": agent_id,
        "authorizePath": "/api/authorize",
        "metadata": metadata,
    }

    # Use simplified chat.js with official Layercode SDK
    audio_script = "/static/chat-frontend.js"

    # Compact status bar with grid layout for better spacing
    status_bar = fasthtml.common.Div(
        fasthtml.common.Div(
            Label(
                "Disconnected",
                id="status-label",
                cls=LabelT.destructive + " text-xs px-2 py-1 col-span-2",
            ),
            Button(
                "Connect",
                id="connect-btn",
                cls=ButtonT.primary
                + " "
                + ButtonT.xs
                + " rounded-full px-4 py-1 col-span-2",
            ),
            Button(
                "Mute Mic",
                id="mute-btn",
                cls=ButtonT.destructive
                + " "
                + ButtonT.xs
                + " rounded-full px-3 py-1 col-span-2",
                disabled=True,
            ),
            cls="grid grid-cols-6 gap-4 items-center",
        ),
        cls="mb-6",
    )

    # Subtle instructions
    instructions = P(
        "Click Connect, then speak or type a message below",
        cls=TextPresets.muted_sm + " mb-3 text-center",
    )

    # Message list with subtle card
    message_card = CardContainer(
        CardBody(
            fasthtml.common.Div(
                id="message-list",
                cls="h-96 overflow-y-auto space-y-2 p-2",
            ),
        ),
        cls="mb-4 shadow-sm",
    )

    # Compact text input with tiny rounded send button (grid 5:1 ratio)
    text_form = Form(
        fasthtml.common.Div(
            Input(
                id="text-input",
                name="message",
                type="text",
                placeholder="Type a message…",
                autocomplete="off",
                cls="col-span-5",
            ),
            Button(
                UkIcon("send", height=14),
                type="submit",
                cls=ButtonT.primary
                + " "
                + ButtonT.icon
                + " col-span-1 h-10 w-10 rounded-full p-0 flex items-center justify-center",
            ),
            cls="grid grid-cols-6 gap-2 items-center",
        ),
        id="text-form",
    )

    # Error display
    error_box = fasthtml.common.Div(
        id="error-box",
        cls="mt-2 text-sm text-destructive hidden",
    )

    # Compact header
    header = fasthtml.common.Div(
        Container(
            DivLAligned(
                UkIcon("mic", height=20, cls="text-primary"),
                fasthtml.common.H1("Layercode Voice Chat", cls="text-lg font-medium"),
                cls="gap-2 py-3",
            ),
            cls=ContainerT.lg,
        ),
        cls="border-b border-border mb-6",
    )

    # Main content layout
    return (
        # Config data for JavaScript
        Script(
            json.dumps(config_payload), type="application/json", id="layercode-config"
        ),
        # Chat JavaScript (WebSocket client with microphone support)
        Script(src=audio_script, type="module"),
        # Page structure
        Title("Layercode Voice Chat"),
        header,
        Container(
            fasthtml.common.Div(
                status_bar,
                instructions,
                message_card,
                text_form,
                error_box,
                cls="py-4",
            ),
            cls=ContainerT.lg,
        ),
    )


# ============================================================================
# Static Files
# ============================================================================

# Mount static files directory if it exists
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.warning(
        f"Static directory '{STATIC_DIR}' missing; frontend assets may not load"
    )


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Run the FastHTML application with uvicorn."""
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(int(os.getenv("UVICORN_RELOAD", "0"))),
    )


if __name__ == "__main__":
    main()
