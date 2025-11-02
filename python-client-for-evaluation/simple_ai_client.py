#!/usr/bin/env python3
"""
LayerCode WebSocket client - AI-powered multi-turn conversations with observability.

Simulates realistic customer interactions using PydanticAI to generate contextual responses.
Fully instrumented with Logfire for debugging and performance analysis.

WORKFLOW:
    1. POST to backend /api/authorize with agent_id
    2. Connect to LayerCode WebSocket
    3. Send client.ready event
    4. Wait for assistant turn (response.text + response.audio events)
    5. Buffer assistant text and audio
    6. When turn.start (role=user) fires:
       a. Finalize assistant turn data
       b. Pass assistant text to PydanticAI customer agent (gpt-5-nano)
       c. Generate contextual reply based on conversation history
       d. Synthesize reply via OpenAI TTS
       e. Stream audio back as client.audio chunks
    7. Repeat for max_user_turns (default: 3)
    8. Wait for final assistant response, then shutdown

AI CUSTOMER PERSONA:
    - Name: Configurable (default: "Jan")
    - Behavior: Polite shopper looking for a blazer
    - Model: OpenAI gpt-5-nano via PydanticAI
    - Memory: Full conversation history maintained

OBSERVABILITY:
    - Logfire integration for PydanticAI + OpenAI traces
    - Request/response logging with token counts
    - Audio buffer timing and sizes
    - Turn-level event tracking

REQUIREMENTS:
    - Backend server at SERVER_URL with /api/authorize endpoint
    - OPENAI_API_KEY for TTS + AI agent
    - LOGFIRE_TOKEN (optional, for cloud tracing)
    - LAYERCODE_AGENT_ID

USAGE:
    export OPENAI_API_KEY="sk-..."
    export LOGFIRE_TOKEN="pylf_..."  # Optional
    python simple_ai_client.py --agent-id YOUR_AGENT_ID --max-user-turns 5
    python simple_ai_client.py --customer-name "Alice" --assistant-idle-timeout 5.0

OUTPUTS:
    - audio/input_TIMESTAMP.wav - AI-generated user audio (8kHz mono PCM)
    - audio/output_TIMESTAMP.wav - Assistant responses (16kHz mono PCM)
    - Logfire traces (if token configured)

NOTE:
    Script does NOT auto-terminate after sending all user turns. It waits for the
    final assistant response before shutting down. For batch evaluation, modify
    shutdown logic at simple_ai_client.py:525-527 to exit immediately after
    max_user_turns is reached.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import base64
import io
import json
import os
import signal
import sys
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from loguru import logger
import logfire
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
from pydantic_ai import Agent
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

load_dotenv()

logfire.configure(
    scrubbing=False,
    service_name="client",
    send_to_logfire="if-token-present",
    environment="development",
)
logfire.instrument_pydantic_ai()
logfire.instrument_openai()

DEFAULT_AUTHORIZE_PATH = "/api/authorize"
DEFAULT_WS_URL = "wss://api.layercode.com/v1/agents/web/websocket"
TARGET_SAMPLE_RATE = 8000


def _utc_timestamp() -> datetime:
    return datetime.now(timezone.utc)


def _format_ts(ts: datetime) -> str:
    return ts.strftime("%Y%m%d-%H%M%S-%f")


@dataclass
class ClientConfig:
    server_url: str
    authorize_path: str = DEFAULT_AUTHORIZE_PATH
    ws_url: str = DEFAULT_WS_URL
    agent_id: str = ""
    audio_dir: Path = Path("audio")
    chunk_ms: int = 100
    chunk_interval: float = 0.0
    log_level: str = "INFO"
    send_ready_delay: float = 0.0
    assistant_idle_timeout: float = 3.0
    debug: bool = True
    max_user_turns: int = 3
    default_reply: str = os.getenv("DEFAULT_REPLY", "Hey. I'm Jan - what's your name?")
    tts_model: str = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    tts_voice: str = os.getenv("OPENAI_TTS_VOICE", "coral")
    tts_instructions: Optional[str] = os.getenv("OPENAI_TTS_INSTRUCTIONS")
    customer_name: str = os.getenv("CUSTOMER_NAME", "Jan")


@dataclass
class AssistantTurnBuffer:
    turn_id: str
    started_at: datetime = field(default_factory=_utc_timestamp)
    sample_rate: int = 16000
    buffer: bytearray = field(default_factory=bytearray)
    last_delta_id: Optional[str] = None
    last_ack_delta_id: Optional[str] = None
    acked_none: bool = False

    def extend(self, chunk: bytes, delta_id: Optional[str]) -> None:
        self.buffer.extend(chunk)
        if delta_id:
            self.last_delta_id = delta_id

    def duration_seconds(self) -> float:
        if not self.buffer:
            return 0.0
        frame_count = len(self.buffer) // 2  # 16-bit PCM -> 2 bytes per sample
        return frame_count / float(self.sample_rate)


class AssistantTurn(BaseModel):
    text: str
    audio_path: Path | None = None


class UserReply(BaseModel):
    transcript: str
    pcm: bytes
    sample_rate: int


class ConversationAgent:
    """Maintains conversation history (user/assistant) and produces polite customer replies."""

    def __init__(self, customer_name: str, default_reply: str, max_turns: int) -> None:
        self.customer_name = customer_name
        self.default_reply = default_reply.strip()
        self.max_turns = max_turns
        self.turns_used = 0
        self.history: list[
            tuple[str, str]
        ] = []  # role -> content, roles: "user" (assistant), "assistant" (customer)
        self.pending_user_message: Optional[str] = None
        instructions = (
            "You are a very polite shopper named "
            f"{customer_name}. You are looking for a simple blazer in a fashion store. "
            "Keep responses concise (1-2 sentences), express gratitude, and stay on topic."
        )
        self.agent = Agent("openai:gpt-5-nano", instructions=instructions)

    def record_assistant(self, text: str) -> None:
        self.pending_user_message = text.strip()

    async def next_reply(self) -> Optional[str]:
        if self.turns_used >= self.max_turns:
            return None
        if not self.pending_user_message:
            return None

        user_message = self.pending_user_message
        self.pending_user_message = None

        history_copy = list(self.history)
        try:
            result = await self.agent.run(user_message, message_history=history_copy)
        except Exception as exc:  # pragma: no cover - protective guard
            raise RuntimeError(
                f"Customer agent failed to generate a reply: {exc}"
            ) from exc

        reply = result.output.strip()
        if not reply:
            raise RuntimeError("Conversation agent returned empty reply")

        # Update conversation history
        self.history = result.all_messages()

        self.turns_used += 1
        return reply


class OpenAITTSVoice:
    """Streams OpenAI TTS output and normalizes it to 16-bit mono PCM at 8 kHz."""

    def __init__(
        self, client: AsyncOpenAI, model: str, voice: str, instructions: Optional[str]
    ) -> None:
        self._client = client
        self._model = model
        self._voice = voice
        self._instructions = instructions

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        audio_bytes = bytearray()
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "voice": self._voice,
            "input": text,
            "response_format": "wav",
        }
        if self._instructions:
            request_kwargs["instructions"] = self._instructions

        try:
            async with self._client.audio.speech.with_streaming_response.create(
                **request_kwargs
            ) as response:
                async for chunk in response.iter_bytes():
                    audio_bytes.extend(chunk)
        except OpenAIError as exc:
            logger.error("OpenAI TTS request failed: {}", exc)
            raise
        except Exception as exc:  # pragma: no cover - protective guard
            logger.exception("Unexpected error during TTS streaming: {}", exc)
            raise

        if not audio_bytes:
            return b"", TARGET_SAMPLE_RATE

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                pcm = wf.readframes(wf.getnframes())
        except wave.Error as exc:
            logger.error("Failed to parse synthesized WAV: {}", exc)
            raise

        if channels > 1:
            pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
            channels = 1

        if sample_width != 2:
            pcm = audioop.lin2lin(pcm, sample_width, 2)
            sample_width = 2

        if sample_rate != TARGET_SAMPLE_RATE:
            pcm, _ = audioop.ratecv(
                pcm, sample_width, 1, sample_rate, TARGET_SAMPLE_RATE, None
            )
            sample_rate = TARGET_SAMPLE_RATE

        return pcm, sample_rate

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if asyncio.iscoroutinefunction(close):
            await close()  # type: ignore[arg-type]
        elif callable(close):
            close()  # type: ignore[misc]


class FakeCustomer:
    """Glue between the conversation agent and the voice synthesizer."""

    def __init__(self, agent: ConversationAgent, voice: OpenAITTSVoice) -> None:
        self._agent = agent
        self._voice = voice

    async def reply(self, assistant_turn: AssistantTurn) -> Optional[UserReply]:
        self._agent.record_assistant(assistant_turn.text)
        text = await self._agent.next_reply()
        if text is None:
            return None

        pcm, sample_rate = await self._voice.synthesize(text)
        if not pcm:
            return None
        if len(pcm) % 2 != 0:
            pcm = pcm[:-1]  # preserve 16-bit alignment
        return UserReply(transcript=text, pcm=pcm, sample_rate=sample_rate)

    async def close(self) -> None:
        await self._voice.close()


class LayercodeAIClient:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self.session_key: Optional[str] = None
        self.conversation_id: Optional[str] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.ws: Any = None

        self.shutdown_event = asyncio.Event()
        self.user_turn_event = asyncio.Event()
        self._assistant_idle_task: Optional[asyncio.Task[None]] = None

        self.ready_sent = False
        self.last_input_timestamp: Optional[str] = None
        self._last_turn_role: Optional[str] = None

        self.assistant_buffers: dict[str, AssistantTurnBuffer] = {}
        self.current_assistant_turn_id: Optional[str] = None
        self.current_assistant_audio_path: Optional[Path] = None
        self.assistant_text_segments: list[str] = []

        self.awaiting_final_assistant = False

        self._openai = AsyncOpenAI()
        convo_agent = ConversationAgent(
            customer_name=self.config.customer_name,
            default_reply=self.config.default_reply,
            max_turns=self.config.max_user_turns,
        )
        voice = OpenAITTSVoice(
            self._openai,
            self.config.tts_model,
            self.config.tts_voice,
            self.config.tts_instructions,
        )
        self.fake_customer = FakeCustomer(convo_agent, voice)

    async def run(self) -> None:
        self.config.audio_dir.mkdir(parents=True, exist_ok=True)
        timeout = httpx.Timeout(10.0, connect=10.0, read=None)
        async with httpx.AsyncClient(timeout=timeout) as client:
            self.http_client = client
            await self._authorize_session()
        self.http_client = None

        ws_url = self._build_ws_url()
        logger.info("Connecting to Layercode WebSocket: {}", ws_url)
        try:
            async with connect(
                ws_url,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
                open_timeout=20,
            ) as websocket:
                self.ws = websocket
                await asyncio.sleep(self.config.send_ready_delay)
                await self._send_ready()
                receiver_task = asyncio.create_task(
                    self._receive_loop(), name="receive_loop"
                )
                coordinator_task = asyncio.create_task(
                    self._turn_coordinator(), name="turn_coordinator"
                )
                await self._wait_for_shutdown({receiver_task, coordinator_task})
        except (ConnectionClosedOK, ConnectionClosedError) as exc:
            logger.warning("WebSocket connection closed: {}", exc)
        finally:
            self.ws = None
            self._cancel_assistant_idle_timer()
        await self._finalize_assistant_turn()
        await self.fake_customer.close()
        logger.info("Client shutdown complete.")

    async def _wait_for_shutdown(self, tasks: set[asyncio.Task[Any]]) -> None:
        try:
            await self.shutdown_event.wait()
        except asyncio.CancelledError:
            raise
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _authorize_session(self) -> None:
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")
        authorize_url = self._build_authorize_url()
        logger.info(
            "Requesting client_session_key from {} (agent_id={})",
            authorize_url,
            self.config.agent_id,
        )
        try:
            response = await self.http_client.post(
                authorize_url, json={"agent_id": self.config.agent_id}
            )
        except httpx.HTTPError as exc:
            logger.exception("Failed to reach authorization endpoint: {}", exc)
            raise SystemExit(1) from exc
        if response.status_code >= 400:
            logger.error(
                "Authorization endpoint returned {}: {}",
                response.status_code,
                response.text,
            )
            raise SystemExit(1)

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            logger.error("Authorization response is not valid JSON: {}", exc)
            raise SystemExit(1) from exc

        self.session_key = payload.get("client_session_key")
        self.conversation_id = payload.get("conversation_id")
        if not self.session_key:
            logger.error(
                "Authorization response missing client_session_key: {}", payload
            )
            raise SystemExit(1)
        logger.success(
            "Received client_session_key (conversation_id={}): {}",
            self.conversation_id,
            self.session_key,
        )

    async def _send_ready(self) -> None:
        if self.ready_sent:
            return
        await self._send_json({"type": "client.ready"})
        self.ready_sent = True
        logger.info("Sent client.ready")

    async def _turn_coordinator(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                await self.user_turn_event.wait()
            except asyncio.CancelledError:
                break
            self.user_turn_event.clear()

            assistant_turn = await self._finalize_assistant_turn()
            if assistant_turn is None:
                if self.config.debug:
                    logger.debug("No assistant turn data available for reply.")
                continue

            await self._deliver_user_reply(assistant_turn)

    async def _deliver_user_reply(self, assistant_turn: AssistantTurn) -> None:
        reply = await self.fake_customer.reply(assistant_turn)
        if reply is None:
            logger.info(
                "Customer agent has no further replies; awaiting final assistant response."
            )
            self.awaiting_final_assistant = True
            return

        logger.info("[USER] {}", reply.transcript)
        await self._stream_user_reply(reply)

    async def _receive_loop(self) -> None:
        assert self.ws is not None
        logger.info("Started receive loop.")
        while not self.shutdown_event.is_set():
            try:
                raw = await self.ws.recv()
            except ConnectionClosedOK as exc:
                logger.info("WebSocket closed cleanly: {}", exc)
                break
            except ConnectionClosedError as exc:
                logger.warning("WebSocket disconnected with error: {}", exc)
                break
            except (asyncio.CancelledError, RuntimeError):
                raise
            except Exception as exc:  # pragma: no cover - diagnostics
                logger.exception("Error while receiving message: {}", exc)
                break

            if raw is None:
                continue
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning("Ignoring non-UTF8 WebSocket payload.")
                    continue

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Received non-JSON message: {}", raw)
                continue

            await self._handle_server_message(message)

        self.shutdown_event.set()
        logger.info("Receive loop finished.")

    async def _handle_server_message(self, message: dict[str, Any]) -> None:
        msg_type = message.get("type")
        self._log_message("incoming", message)

        if msg_type == "turn.start":
            await self._handle_turn_start(message)
        elif msg_type == "response.audio":
            await self._handle_response_audio(message)
        elif msg_type == "response.data":
            content = message.get("content")
            if self.config.debug:
                logger.info("Received response.data payload: {}", content)
            else:
                compact = (
                    json.dumps(content, separators=(",", ":"))
                    if content is not None
                    else ""
                )
                logger.info("[DATA] {}", compact)
        elif msg_type == "response.text":
            content = (message.get("content") or "").strip()
            if content:
                self.assistant_text_segments.append(content)
                if self.config.debug:
                    logger.info("[ASSISTANT] {}", content)
        elif msg_type in {
            "user.transcript",
            "user.transcript.delta",
            "user.transcript.interim_delta",
        }:
            content = message.get("content")
            if msg_type == "user.transcript":
                logger.info("[USER] {}", content)
            elif self.config.debug:
                logger.debug("Transcript event: {} -> {}", msg_type, content)

    async def _handle_turn_start(self, message: dict[str, Any]) -> None:
        role = message.get("role")
        if self.config.debug:
            logger.info("Turn started for role={}", role)

        if role == "assistant":
            # Flush any lingering assistant content before starting a new turn.
            await self._finalize_assistant_turn()
            self._reset_assistant_state()
        elif role == "user":
            self.user_turn_event.set()
        self._last_turn_role = role

    async def _handle_response_audio(self, message: dict[str, Any]) -> None:
        content = message.get("content")
        turn_id = message.get("turn_id")
        if not isinstance(content, str) or not turn_id:
            logger.warning("Malformed response.audio message: {}", message)
            return

        try:
            decoded = base64.b64decode(content)
        except Exception as exc:
            logger.error("Failed to decode base64 response audio: {}", exc)
            return

        if self.current_assistant_turn_id and self.current_assistant_turn_id != turn_id:
            await self._finalize_assistant_turn()

        buffer = self.assistant_buffers.get(turn_id)
        if buffer is None:
            buffer = AssistantTurnBuffer(turn_id=turn_id)
            self.assistant_buffers[turn_id] = buffer
            logger.info(
                "Started capturing assistant audio for turn {} at {}",
                turn_id,
                _format_ts(buffer.started_at),
            )
        delta_id = message.get("delta_id")
        buffer.extend(decoded, delta_id)
        self.current_assistant_turn_id = turn_id
        await self._acknowledge_audio(turn_id, delta_id)
        self._schedule_assistant_idle_check(turn_id)

    async def _finalize_assistant_turn(self) -> Optional[AssistantTurn]:
        if self.current_assistant_turn_id:
            await self._finalize_assistant_audio(self.current_assistant_turn_id)
            self.current_assistant_turn_id = None

        text = " ".join(
            seg.strip() for seg in self.assistant_text_segments if seg.strip()
        ).strip()
        audio_path = self.current_assistant_audio_path

        if not text and audio_path is None:
            self._reset_assistant_state()
            return None

        if text and not self.config.debug:
            logger.info("[ASSISTANT] {}", text)

        turn = AssistantTurn(text=text, audio_path=audio_path)
        if self.awaiting_final_assistant and text:
            logger.info("Final assistant turn captured; shutting down after response.")
            self.shutdown_event.set()
        self._reset_assistant_state()
        return turn

    def _reset_assistant_state(self) -> None:
        self.assistant_text_segments = []
        self.current_assistant_audio_path = None

    async def _finalize_assistant_audio(self, turn_id: str) -> Optional[Path]:
        buffer = self.assistant_buffers.get(turn_id)
        if buffer is None:
            return None
        if not buffer.buffer:
            logger.debug("Assistant audio buffer for {} is empty; skipping.", turn_id)
            self.assistant_buffers.pop(turn_id, None)
            return None

        timestamp = _format_ts(buffer.started_at)
        output_path = self.config.audio_dir / f"output_{timestamp}.wav"
        try:
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(buffer.sample_rate)
                wf.writeframes(buffer.buffer)
            logger.success(
                "Saved assistant audio turn {} ({}s, {} bytes) to {}",
                turn_id,
                round(buffer.duration_seconds(), 2),
                len(buffer.buffer),
                output_path,
            )
        except Exception as exc:
            logger.exception(
                "Failed to persist assistant audio to {}: {}", output_path, exc
            )

        await self._acknowledge_audio(turn_id, buffer.last_delta_id, force=True)
        self.assistant_buffers.pop(turn_id, None)
        self.current_assistant_audio_path = output_path
        if self.awaiting_final_assistant:
            logger.info("Final assistant audio captured; initiating shutdown.")
            self.shutdown_event.set()
        return output_path

    async def _stream_user_reply(self, reply: UserReply) -> None:
        timestamp = _format_ts(_utc_timestamp())
        self.last_input_timestamp = timestamp
        wav_path = self.config.audio_dir / f"input_{timestamp}.wav"
        try:
            self._write_wav_file(wav_path, reply.pcm, reply.sample_rate)
            logger.info("Saved synthesized user audio to {}", wav_path)
        except Exception as exc:
            logger.warning("Unable to persist synthesized audio: {}", exc)

        frame_bytes = 2  # mono 16-bit PCM
        frames_per_chunk = (
            max(len(reply.pcm) // frame_bytes, 1)
            if self.config.chunk_ms <= 0
            else max(int(reply.sample_rate * (self.config.chunk_ms / 1000.0)), 1)
        )
        chunk_size = frames_per_chunk * frame_bytes

        logger.info(
            "Streaming synthesized reply ({} Hz) in chunks of {} frame(s)",
            reply.sample_rate,
            frames_per_chunk,
        )

        offset = 0
        chunks_sent = 0
        while offset < len(reply.pcm):
            chunk = reply.pcm[offset : offset + chunk_size]
            offset += chunk_size
            if not chunk:
                break
            payload = {
                "type": "client.audio",
                "content": base64.b64encode(chunk).decode("ascii"),
            }
            await self._send_json(payload)
            if self.config.chunk_interval > 0:
                await asyncio.sleep(self.config.chunk_interval)
            chunks_sent += 1

        logger.info(
            "Finished streaming synthesized user audio ({} chunks sent).", chunks_sent
        )

    async def _acknowledge_audio(
        self, turn_id: str, delta_id: Optional[str], force: bool = False
    ) -> None:
        buffer = self.assistant_buffers.get(turn_id)
        if buffer is not None:
            already_acked = (delta_id is None and buffer.acked_none) or (
                delta_id is not None and buffer.last_ack_delta_id == delta_id
            )
            if already_acked and not force:
                return
        elif not force:
            return

        payload: dict[str, Any] = {
            "type": "trigger.response.audio.replay_finished",
            "reason": "completed",
            "turn_id": turn_id,
        }
        if delta_id:
            payload["last_delta_id_played"] = delta_id
        await self._send_json(payload)
        if self.config.debug:
            logger.info(
                "Acknowledged assistant audio turn {} delta {} (force={})",
                turn_id,
                delta_id,
                force,
            )
        if buffer is not None:
            buffer.last_ack_delta_id = delta_id
            if delta_id is None:
                buffer.acked_none = True

    def _schedule_assistant_idle_check(self, turn_id: str) -> None:
        self._cancel_assistant_idle_timer()
        timeout = self.config.assistant_idle_timeout
        if timeout <= 0:
            return

        async def idle_timer() -> None:
            try:
                await asyncio.sleep(timeout)
                if (
                    self.current_assistant_turn_id == turn_id
                    and not self.shutdown_event.is_set()
                ):
                    logger.info(
                        "Assistant idle for {:.2f}s on turn {}; triggering user reply.",
                        timeout,
                        turn_id,
                    )
                    self.user_turn_event.set()
            except asyncio.CancelledError:
                return

        loop = asyncio.get_running_loop()
        self._assistant_idle_task = loop.create_task(
            idle_timer(), name="assistant_idle_timer"
        )

    def _cancel_assistant_idle_timer(self) -> None:
        if self._assistant_idle_task and not self._assistant_idle_task.done():
            self._assistant_idle_task.cancel()
        self._assistant_idle_task = None

    def _write_wav_file(self, path: Path, pcm: bytes, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if not self.ws:
            raise RuntimeError("WebSocket not connected")
        self._log_message("outgoing", payload)
        try:
            await self.ws.send(json.dumps(payload, separators=(",", ":")))
        except Exception as exc:
            logger.exception("Failed to send WebSocket payload {}: {}", payload, exc)

    def _log_message(self, direction: str, payload: dict[str, Any]) -> None:
        if not self.config.debug:
            return
        msg_type = payload.get("type")
        printable = dict(payload)
        if msg_type in {"client.audio", "response.audio"} and "content" in printable:
            content = printable.pop("content")
            printable["content_length"] = len(content)
        logger.bind(direction=direction, event_type=msg_type).info(
            "{} {}", direction, printable
        )

    def _build_authorize_url(self) -> str:
        base = self.config.server_url.rstrip("/")
        path = self.config.authorize_path.lstrip("/")
        return f"{base}/{path}"

    def _build_ws_url(self) -> str:
        if not self.session_key:
            raise RuntimeError("Session key not available")
        separator = "&" if "?" in self.config.ws_url else "?"
        return f"{self.config.ws_url}{separator}client_session_key={self.session_key}"


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=level.upper(),
        backtrace=False,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
    )


def _parse_args() -> ClientConfig:
    parser = argparse.ArgumentParser(
        description="Layercode WebSocket client that runs a 3-turn AI customer conversation"
    )
    parser.add_argument(
        "--server-url",
        default=os.getenv("SERVER_URL", "http://localhost:8001"),
        help="Backend base URL (default: $SERVER_URL or http://localhost:8001)",
    )
    parser.add_argument(
        "--authorize-path",
        default=DEFAULT_AUTHORIZE_PATH,
        help="Relative path for the backend authorization route (default: /session/authorize)",
    )
    parser.add_argument(
        "--ws-url",
        default=DEFAULT_WS_URL,
        help="Layercode WebSocket endpoint (default: wss://api.layercode.com/v1/agents/web/websocket)",
    )
    parser.add_argument(
        "--agent-id",
        default=os.getenv("LAYERCODE_AGENT_ID"),
        help="Layercode agent identifier (default: $LAYERCODE_AGENT_ID)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Frames per chunk duration in milliseconds",
    )
    parser.add_argument(
        "--chunk-interval",
        type=float,
        default=0.0,
        help="Pause between sending audio chunks in seconds",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Log level for console output (default: INFO or $LOG_LEVEL)",
    )
    parser.add_argument(
        "--send-ready-delay",
        type=float,
        default=0.0,
        help="Delay in seconds before sending client.ready after the WebSocket opens",
    )
    parser.add_argument(
        "--assistant-idle-timeout",
        type=float,
        default=3.0,
        help="Seconds of silence after assistant audio before starting user turn",
    )
    parser.add_argument(
        "--max-user-turns",
        type=int,
        default=3,
        help="Maximum number of customer turns before closing the session",
    )
    parser.add_argument(
        "--default-reply",
        default=os.getenv("DEFAULT_REPLY", "Hey. I'm Jan - what's your name?"),
        help="Fallback text to synthesize when the agent yields nothing",
    )
    parser.add_argument(
        "--tts-model",
        default=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        help="OpenAI TTS model to use",
    )
    parser.add_argument(
        "--tts-voice",
        default=os.getenv("OPENAI_TTS_VOICE", "coral"),
        help="OpenAI TTS voice to use",
    )
    parser.add_argument(
        "--tts-instructions",
        default=os.getenv("OPENAI_TTS_INSTRUCTIONS"),
        help="Optional additional instructions for the TTS voice",
    )
    parser.add_argument(
        "--customer-name",
        default=os.getenv("CUSTOMER_NAME", "Jan"),
        help="Name the customer agent should use",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=True,
        help="Enable verbose event logging (default: on)",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Minimal logging: transcripts and audio summaries only",
    )
    args = parser.parse_args()

    agent_id = args.agent_id
    if not agent_id:
        raise SystemExit(
            "Agent id missing. Provide --agent-id or set LAYERCODE_AGENT_ID."
        )

    return ClientConfig(
        server_url=args.server_url,
        authorize_path=args.authorize_path,
        ws_url=args.ws_url,
        agent_id=agent_id,
        chunk_ms=args.chunk_ms,
        chunk_interval=args.chunk_interval,
        log_level=args.log_level,
        send_ready_delay=args.send_ready_delay,
        assistant_idle_timeout=args.assistant_idle_timeout,
        debug=args.debug,
        max_user_turns=args.max_user_turns,
        default_reply=args.default_reply,
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        tts_instructions=args.tts_instructions,
        customer_name=args.customer_name,
    )


async def _async_main(config: ClientConfig) -> None:
    client = LayercodeAIClient(config)
    loop = asyncio.get_running_loop()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue

        try:
            loop.add_signal_handler(sig, client.shutdown_event.set)
        except NotImplementedError:
            pass

    try:
        await client.run()
    except asyncio.CancelledError:
        logger.info("Cancelled.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


def main() -> None:
    load_dotenv()
    config = _parse_args()
    _configure_logging(config.log_level)
    logger.info("Starting Layercode AI client with config: {}", config)
    try:
        asyncio.run(_async_main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted. Exiting.")


if __name__ == "__main__":
    main()
