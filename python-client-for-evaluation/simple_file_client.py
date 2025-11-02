#!/usr/bin/env python3
"""
Simple Layercode WebSocket file-based client.

This script emulates the browser Layercode SDK for local development:
1. Authorizes a client session by calling the backend at SERVER_URL/session/authorized.
2. Opens the Layercode frontend WebSocket with the returned client_session_key.
3. Waits for the server to hand the turn to the user, then streams a WAV file as audio chunks.
4. Logs inbound/outbound events, persisting assistant audio responses under audio/output_*.

Usage:
    python simple_file_client.py --server-url http://localhost:8001 --agent-id YOUR_AGENT
    # or export LAYERCODE_AGENT_ID before running
"""

from __future__ import annotations

import argparse
import asyncio
import base64
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
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

DEFAULT_AUTHORIZE_PATH = "/api/authorize"
DEFAULT_WS_URL = "wss://api.layercode.com/v1/agents/web/websocket"


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
    audio_input: Path = Path("audio/intro-example-8000hz.wav")
    audio_dir: Path = Path("audio")
    chunk_ms: int = 100
    chunk_interval: float = 0.0
    log_level: str = "INFO"
    send_ready_delay: float = 0.0
    auto_send_once: bool = True
    assistant_idle_timeout: float = 3.0
    debug: bool = True


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


class LayercodeFileClient:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self.session_key: Optional[str] = None
        self.conversation_id: Optional[str] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.ws: Any = None
        self.shutdown_event = asyncio.Event()
        self.user_turn_event = asyncio.Event()
        self.audio_sent = False
        self.ready_sent = False
        self.last_input_timestamp: Optional[str] = None
        self.assistant_buffers: dict[str, AssistantTurnBuffer] = {}
        self.current_assistant_turn_id: Optional[str] = None
        self._last_turn_role: Optional[str] = None
        self._assistant_idle_task: Optional[asyncio.Task[None]] = None

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
            logger.warning("WebSocket connection closed during setup: {}", exc)
        finally:
            self.ws = None
            self._cancel_assistant_idle_timer()
        await self._finalize_current_assistant_audio()
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
                authorize_url,
                json={"agent_id": self.config.agent_id},
            )
        except httpx.HTTPError as exc:
            logger.exception("Failed to reach backend authorization endpoint: {}", exc)
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
            if self.audio_sent and not self.config.auto_send_once:
                logger.debug(
                    "User turn signaled but auto_send_once is disabled and audio already sent."
                )
                continue
            if self.audio_sent and self.config.auto_send_once:
                logger.info("User turn signaled but audio was already sent; ignoring.")
                continue
            logger.info("User turn granted; streaming {}.", self.config.audio_input)
            await self._stream_audio_file()
            self.audio_sent = True

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
            logger.info("[ASSISTANT] {}", message.get("content"))
        elif msg_type in {
            "user.transcript",
            "user.transcript.delta",
            "user.transcript.interim_delta",
        }:
            content = message.get("content")
            if msg_type == "user.transcript":
                logger.info("[USER] {}", content)
            else:
                if self.config.debug:
                    logger.debug("Transcript event: {} -> {}", msg_type, content)

    async def _handle_turn_start(self, message: dict[str, Any]) -> None:
        role = message.get("role")
        if self.config.debug:
            logger.info("Turn started for role={}", role)

        if role == "assistant":
            self._cancel_assistant_idle_timer()
            await self._finalize_current_assistant_audio()
        elif role == "user":
            self._cancel_assistant_idle_timer()
            await self._finalize_current_assistant_audio()
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
            await self._finalize_current_assistant_audio()

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

    async def _stream_audio_file(self) -> None:
        audio_path = self.config.audio_input
        if not audio_path.exists():
            logger.error("Audio file not found: {}", audio_path)
            return

        timestamp = _format_ts(_utc_timestamp())
        self.last_input_timestamp = timestamp
        mirrored_path = self.config.audio_dir / f"input_{timestamp}.wav"
        try:
            mirrored_path.write_bytes(audio_path.read_bytes())
            logger.info("Mirrored outbound audio to {}", mirrored_path)
        except Exception as exc:
            logger.warning("Unable to mirror outbound audio: {}", exc)

        try:
            with wave.open(str(audio_path), "rb") as wf:
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                if sample_width != 2 or channels != 1:
                    logger.warning(
                        "Audio file {} is {} channel(s) at {} bytes/sample. Layercode expects mono 16-bit PCM. "
                        "Proceeding anyway.",
                        audio_path,
                        channels,
                        sample_width,
                    )
                frames_per_chunk = (
                    wf.getnframes()
                    if self.config.chunk_ms <= 0
                    else max(int(sample_rate * (self.config.chunk_ms / 1000.0)), 1)
                )
                logger.info(
                    "Streaming audio {} ({} Hz) in chunks of {} frame(s)",
                    audio_path,
                    sample_rate,
                    frames_per_chunk,
                )
                while True:
                    frames = wf.readframes(frames_per_chunk)
                    if not frames:
                        break
                    payload = {
                        "type": "client.audio",
                        "content": base64.b64encode(frames).decode("ascii"),
                    }
                    await self._send_json(payload)
                    if self.config.chunk_interval > 0:
                        await asyncio.sleep(self.config.chunk_interval)

            logger.info("Finished streaming user audio.")
        except FileNotFoundError:
            logger.error("Audio file not found: {}", audio_path)
        except wave.Error as exc:
            logger.error("Failed to read WAV file {}: {}", audio_path, exc)
        except Exception as exc:
            logger.exception("Unexpected error while streaming audio: {}", exc)

    async def _finalize_current_assistant_audio(self, reason: str = "manual") -> None:
        self._cancel_assistant_idle_timer()
        if not self.current_assistant_turn_id:
            return
        turn_id = self.current_assistant_turn_id
        await self._finalize_assistant_audio(turn_id, reason=reason)
        self.current_assistant_turn_id = None

    async def _finalize_assistant_audio(
        self, turn_id: str, reason: str = "manual"
    ) -> bool:
        buffer = self.assistant_buffers.get(turn_id)
        if buffer is None:
            return False
        if not buffer.buffer:
            logger.debug("Assistant audio buffer for {} is empty; skipping.", turn_id)
            return False

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
        # Remove buffer after ack to release memory.
        self.assistant_buffers.pop(turn_id, None)
        logger.info(
            "Finalized assistant audio turn {} (reason={}) -> {}",
            turn_id,
            reason,
            output_path,
        )
        return True

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
                if self.current_assistant_turn_id == turn_id:
                    logger.info(
                        "Assistant idle for {:.2f}s on turn {}; finalizing.",
                        timeout,
                        turn_id,
                    )
                    finalized = await self._finalize_assistant_audio(
                        turn_id, reason="idle_timeout"
                    )
                    if finalized:
                        self.current_assistant_turn_id = None
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
        description="Layercode WebSocket simple file client"
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
        "--audio-input",
        default="audio/intro-example-8000hz.wav",
        help="Path to WAV file to stream on the first user turn (default: audio/intro-example-8000hz.wav)",
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
        help="Frames per chunk duration in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--chunk-interval",
        type=float,
        default=0.0,
        help="Pause between sending audio chunks in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Log level for console output (default: INFO or $LOG_LEVEL)",
    )
    parser.add_argument(
        "--no-auto-send",
        action="store_true",
        help="If set, do not automatically send the audio file after the first user turn.",
    )
    parser.add_argument(
        "--send-ready-delay",
        type=float,
        default=0.0,
        help="Delay in seconds before sending client.ready after the WebSocket opens (default: 0.0)",
    )
    parser.add_argument(
        "--assistant-idle-timeout",
        type=float,
        default=3.0,
        help="Seconds of silence after assistant audio before starting user turn (default: 3.0)",
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
        audio_input=Path(args.audio_input),
        chunk_ms=args.chunk_ms,
        chunk_interval=args.chunk_interval,
        log_level=args.log_level,
        auto_send_once=not args.no_auto_send,
        send_ready_delay=args.send_ready_delay,
        assistant_idle_timeout=args.assistant_idle_timeout,
        debug=args.debug,
    )


async def _async_main(config: ClientConfig) -> None:
    client = LayercodeFileClient(config)
    loop = asyncio.get_running_loop()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue

        try:
            loop.add_signal_handler(sig, client.shutdown_event.set)
        except NotImplementedError:
            # Signal handling not supported (e.g., on Windows with Proactor loop)
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
    logger.info("Starting Layercode file client with config: {}", config)
    try:
        asyncio.run(_async_main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted. Exiting.")


if __name__ == "__main__":
    main()
