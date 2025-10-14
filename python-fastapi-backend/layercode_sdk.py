"""
Layercode Python SDK (unofficial)
Utilities for building voice agents with Layercode.
"""

import hashlib
import hmac
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from fastapi.responses import StreamingResponse


class Encoder(Protocol):
    """Protocol for encoders."""

    def encode(self, text: str) -> bytes:
        """Encode text to bytes."""
        ...


class Controller(Protocol):
    """Protocol for stream controllers."""

    def enqueue(self, data: bytes) -> bytes | None:
        """Enqueue data to the stream."""
        ...

    def close(self) -> None:
        """Close the stream."""
        ...


def verify_signature(
    payload: str, signature: str, secret: str, tolerance_seconds: int = 300
) -> bool:
    """
    Verify a Layercode webhook signature.

    Args:
        payload: The raw JSON request body as a string
        signature: The value of the 'layercode-signature' header
        secret: Your LAYERCODE_WEBHOOK_SECRET
        tolerance_seconds: Maximum age of the signature (default: 300 seconds)

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # Parse signature header: "t=<timestamp>,v1=<signature>"
        parts = dict(item.split("=", 1) for item in signature.split(","))
        timestamp = parts["t"]
        sig = parts["v1"]

        # Verify timestamp is not too old
        timestamp_int = int(timestamp)
        current_time = int(time.time())
        if abs(current_time - timestamp_int) > tolerance_seconds:
            return False

        # Compute expected signature
        signed_payload = timestamp.encode("utf-8") + b"." + payload.encode("utf-8")
        expected_signature = hmac.new(
            secret.encode("utf-8"), signed_payload, hashlib.sha256
        ).hexdigest()

        # Compare signatures
        return hmac.compare_digest(expected_signature, sig)
    except Exception:
        return False


class StreamHelper:
    """
    Helper class for building Layercode SSE responses.
    """

    def __init__(self, turn_id: str, encoder: Encoder, controller: Controller) -> None:
        self.turn_id = turn_id
        self._encoder = encoder
        self._controller = controller

    def _send_event(self, event_type: str, content: dict[str, Any]) -> None:
        """Send an SSE event."""
        payload = {"type": event_type, "turn_id": self.turn_id, **content}
        sse = f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
        self._controller.enqueue(self._encoder.encode(sse))

    def tts(self, content: str) -> None:
        """Send a text-to-speech chunk."""
        self._send_event("response.tts", {"content": content})

    async def tts_text_stream(self, text_stream: AsyncIterator[str]) -> None:
        """Stream text chunks as TTS events."""
        async for chunk in text_stream:
            if chunk:
                self.tts(chunk)

    def data(self, content: dict[str, Any]) -> None:
        """Send a data event (for metadata or custom content)."""
        self._send_event("response.data", {"content": content})

    def end(self) -> None:
        """Send the response.end event and close the stream."""
        self._send_event("response.end", {})
        self._controller.close()


async def stream_response(
    request_body: dict[str, Any], handler: Callable[[StreamHelper], Awaitable[None]]
) -> StreamingResponse:
    """
    Creates a server-sent events (SSE) stream response for Layercode.

    Args:
        request_body: The webhook request body (parsed JSON dict)
        handler: Async function that receives a StreamHelper for writing to the stream

    Returns:
        StreamingResponse configured for SSE

    Example:
        @app.post("/api/agent")
        async def agent_webhook(request: Request):
            body = await request.json()

            return await stream_response(body, async (stream) => {
                stream.tts("Hello!")
                stream.end()
            })
    """
    turn_id = request_body.get("turn_id", "")

    class UTF8Encoder:
        def encode(self, text: str) -> bytes:
            return text.encode("utf-8")

    utf8_encoder = UTF8Encoder()
    stream_controller = None

    async def generate() -> None:
        nonlocal stream_controller

        class Controller:
            def __init__(self) -> None:
                self._closed = False

            def enqueue(self, data: bytes) -> bytes | None:
                if not self._closed:
                    return data
                return None

            def close(self) -> None:
                self._closed = True

        stream_controller = Controller()
        stream = StreamHelper(turn_id, utf8_encoder, stream_controller)

        try:
            # Call the user's handler
            await handler(stream)

            # If handler didn't call end(), call it now
            if not stream_controller._closed:
                stream.end()
        except Exception as exc:
            # Send error event
            if not stream_controller._closed:
                stream.data({"error": str(exc)})
                stream.end()

    # Create async generator that yields bytes
    async def stream_generator() -> AsyncIterator[bytes]:
        controller_items: list[bytes] = []

        class YieldController:
            def enqueue(self, data: bytes) -> None:
                controller_items.append(data)

            def close(self) -> None:
                pass

        yield_controller = YieldController()
        stream = StreamHelper(turn_id, utf8_encoder, yield_controller)

        try:
            await handler(stream)

            # Yield all accumulated items
            for item in controller_items:
                yield item

            # Send end event if not already sent
            end_event = f"data: {json.dumps({'type': 'response.end', 'turn_id': turn_id}, separators=(',', ':'))}\n\n"
            yield end_event.encode("utf-8")

        except Exception as exc:
            # Yield any accumulated items first
            for item in controller_items:
                yield item

            # Send error
            error_payload = {
                "type": "response.data",
                "turn_id": turn_id,
                "content": {"error": str(exc)},
            }
            error_sse = f"data: {json.dumps(error_payload, separators=(',', ':'))}\n\n"
            yield error_sse.encode("utf-8")

            # Send end
            end_event = f"data: {json.dumps({'type': 'response.end', 'turn_id': turn_id}, separators=(',', ':'))}\n\n"
            yield end_event.encode("utf-8")

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "Content-Encoding": "none",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        stream_generator(),
        headers=headers,
        media_type="text/event-stream",
    )
