from typing import TypedDict
import asyncio
import base64
import json
from enum import Enum
from pydantic import Field
from nodetool.metadata.types import Chunk, AudioRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.environment import Environment
from nodetool.workflows.io import NodeInputs, NodeOutputs
import logging
from io import BytesIO
import aiohttp

log = logging.getLogger(__name__)


class AudioFormatEnum(str, Enum):
    """Audio format options for speech-to-text."""

    PCM_8000 = "pcm_8000"
    PCM_16000 = "pcm_16000"
    PCM_22050 = "pcm_22050"
    PCM_24000 = "pcm_24000"
    PCM_44100 = "pcm_44100"
    PCM_48000 = "pcm_48000"
    ULAW_8000 = "ulaw_8000"


class CommitStrategyEnum(str, Enum):
    """Strategy for committing transcriptions."""

    MANUAL = "manual"
    VAD = "vad"


class ModelIDEnum(str, Enum):
    """Available ElevenLabs speech-to-text models."""

    SCRIBE_V1 = "scribe_v1"
    SCRIBE_V2 = "scribe_v2"


class TimestampsGranularityEnum(str, Enum):
    """Granularity of timestamps in transcription."""

    NONE = "none"
    WORD = "word"
    CHARACTER = "character"


class FileFormatEnum(str, Enum):
    """File format options for batch transcription."""

    PCM_S16LE_16 = "pcm_s16le_16"  # 16-bit PCM at 16kHz, lower latency
    OTHER = "other"  # All other formats


class SpeechToText(BaseNode):
    """
    Transcribe audio or video files using ElevenLabs speech-to-text API.
    audio, transcription, speech-to-text, stt
    
    Use cases:
    - Transcribe audio files
    - Extract text from video
    - Speaker diarization
    - Multi-language transcription
    - Subtitle generation
    """

    audio: AudioRef = Field(
        default=AudioRef(),
        description="The audio or video file to transcribe"
    )
    model_id: ModelIDEnum = Field(
        default=ModelIDEnum.SCRIBE_V2,
        description="The transcription model to use"
    )
    language_code: str = Field(
        default="",
        description="ISO-639-1 or ISO-639-3 language code (e.g., 'en', 'es'). Leave empty for auto-detection."
    )
    tag_audio_events: bool = Field(
        default=True,
        description="Tag audio events like (laughter), (footsteps), etc."
    )
    num_speakers: int = Field(
        default=0,
        ge=0,
        le=32,
        description="Maximum number of speakers (0 for automatic detection, max 32)"
    )
    timestamps_granularity: TimestampsGranularityEnum = Field(
        default=TimestampsGranularityEnum.WORD,
        description="Granularity of timestamps: none, word, or character"
    )
    diarize: bool = Field(
        default=False,
        description="Annotate which speaker is talking"
    )
    file_format: FileFormatEnum = Field(
        default=FileFormatEnum.OTHER,
        description="Audio format: pcm_s16le_16 for lower latency or other for all formats"
    )

    async def process(self, context: ProcessingContext) -> dict:
        """Transcribe audio file and return transcript."""
        api_key = await context.get_secret_required("ELEVENLABS_API_KEY")
        
        # Get audio data
        audio_data = await context.asset_to_io(self.audio)
        audio_bytes = audio_data.read()
        
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        
        # Build form data
        form_data = aiohttp.FormData()
        form_data.add_field("model_id", self.model_id.value)
        form_data.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
        
        if self.language_code:
            form_data.add_field("language_code", self.language_code)
        
        form_data.add_field("tag_audio_events", str(self.tag_audio_events).lower())
        
        if self.num_speakers > 0:
            form_data.add_field("num_speakers", str(self.num_speakers))
        
        form_data.add_field("timestamps_granularity", self.timestamps_granularity.value)
        form_data.add_field("diarize", str(self.diarize).lower())
        form_data.add_field("file_format", self.file_format.value)
        
        headers = {"xi-api-key": api_key}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form_data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"ElevenLabs API error: {error_text}")
                
                result = await response.json()
                
                # Return the transcript data
                return {
                    "text": result.get("text", ""),
                    "language_code": result.get("language_code", ""),
                    "language_probability": result.get("language_probability", 0.0),
                    "words": result.get("words", []),
                    "transcription_id": result.get("transcription_id"),
                }


class RealtimeSpeechToText(BaseNode):
    """
    (WIP) Realtime speech-to-text transcription using ElevenLabs WebSocket API.
    
    Streams audio chunks in and receives transcription results in real-time.
    Supports both manual commit and voice activity detection (VAD) modes.
    
    Use this node with:
    - model_id: The transcription model to use
    - audio_format: Input audio format (PCM16 recommended)
    - commit_strategy: "manual" or "vad" (voice activity detection)
    - include_timestamps: Get word-level timing information
    - include_language_detection: Detect the spoken language
    """

    chunk: Chunk = Field(
        default=Chunk(),
        description="Audio chunk input stream. Expects base64-encoded audio data with metadata.",
    )
    model_id: ModelIDEnum = Field(
        default=ModelIDEnum.SCRIBE_V2,
        description="Model ID to use for transcription.",
    )
    language_code: str = Field(
        default="",
        description="Language code (ISO 639-1 or ISO 639-3). Leave empty for auto-detection.",
    )
    commit_strategy: CommitStrategyEnum = Field(
        default=CommitStrategyEnum.VAD,
        description="Strategy for committing transcriptions: manual or voice activity detection (VAD).",
    )
    include_timestamps: bool = Field(
        default=False,
        description="Include word-level timestamps in the transcription.",
    )
    include_language_detection: bool = Field(
        default=False,
        description="Include language detection in the transcription.",
    )
    vad_silence_threshold_secs: float = Field(
        default=1.5,
        description="Silence threshold in seconds for VAD mode.",
    )
    vad_threshold: float = Field(
        default=0.4,
        description="Threshold for voice activity detection.",
    )
    min_speech_duration_ms: int = Field(
        default=100,
        description="Minimum speech duration in milliseconds.",
    )
    min_silence_duration_ms: int = Field(
        default=100,
        description="Minimum silence duration in milliseconds.",
    )

    class OutputType(TypedDict):
        chunk: Chunk

    @classmethod
    def return_type(cls):
        return cls.OutputType

    async def run(
        self,
        context: ProcessingContext,
        inputs: NodeInputs,
        outputs: NodeOutputs,
    ) -> None:
        """
        Process audio chunks and stream transcription results.

        Args:
            context (ProcessingContext): The processing context.
            outputs (NodeOutputs): Output emitter for streaming data.
        """
        import websockets

        api_key = await context.get_secret_required("ELEVENLABS_API_KEY")

        # Build WebSocket URL with query parameters
        base_url = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
        params = {
            "model_id": self.model_id.value,
            "file_format": "pcm_s16le_16",
            "commit_strategy": self.commit_strategy.value,
            "include_timestamps": str(self.include_timestamps).lower(),
            "include_language_detection": str(self.include_language_detection).lower(),
            "vad_silence_threshold_secs": str(self.vad_silence_threshold_secs),
            "vad_threshold": str(self.vad_threshold),
            "min_speech_duration_ms": str(self.min_speech_duration_ms),
            "min_silence_duration_ms": str(self.min_silence_duration_ms),
        }
        if self.language_code:
            params["language_code"] = self.language_code

        # Build query string
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        ws_url = f"{base_url}?{query_string}"

        log.info(f"Connecting to ElevenLabs Speech-to-Text WebSocket")

        # Connect to WebSocket with API key in headers
        headers = {"xi-api-key": api_key}

        try:
            async with websockets.connect(ws_url, additional_headers=headers) as websocket:  # type: ignore[arg-type]
                log.info("Connected to ElevenLabs Speech-to-Text WebSocket")

                # Wait for session_started message
                log.debug("Waiting for session_started message")
                first_message = await websocket.recv()
                session_data = json.loads(first_message)
                
                if session_data.get("message_type") != "session_started":
                    raise RuntimeError(f"Expected session_started, got: {session_data.get('message_type')}")
                
                session_id = session_data.get("session_id")
                log.info(f"Session started: {session_id}")
                log.debug(f"Session config: {session_data.get('config')}")

                # Shared state for coordinating producer/consumer
                shared = {"done": False, "session_ready": True}

                # Run producer and consumer loops concurrently
                await asyncio.gather(
                    self._producer_loop(context, websocket, inputs, shared),
                    self._consumer_loop(websocket, outputs, shared),
                )

                log.info("WebSocket session completed successfully")

        except Exception as e:
            log.error(f"WebSocket error: {e}")
            raise RuntimeError(f"Speech-to-text WebSocket failed: {e}")

    async def _producer_loop(
        self,
        context: ProcessingContext,
        websocket,
        inputs: NodeInputs,
        shared: dict,
    ) -> None:
        """Send audio chunks to WebSocket.

        Args:
            context (ProcessingContext): The processing context.
            websocket: WebSocket connection.
            shared (dict): Shared state dict with flags/counters.
        """
        log.debug("Producer loop started")

        # Get sample rate from audio format
        sample_rate = 16000
        first_chunk = True

        async for handle, chunk in inputs.any():
            if handle == "chunk":
                assert isinstance(chunk, Chunk)
                
                if chunk.content_type != "audio":
                    log.warning(f"Skipping non-audio chunk: {chunk.content_type}")
                    continue

                if chunk.done:
                    log.debug("Received done signal, ending producer loop")
                    break

                # Extract base64 audio data
                audio_b64 = chunk.content
                if not audio_b64:
                    log.warning("Received empty audio chunk")
                    continue

                # Validate audio format from metadata
                chunk_metadata = chunk.content_metadata or {}
                chunk_encoding = chunk_metadata.get("encoding", "")
                chunk_sample_rate = chunk_metadata.get("sample_rate", 0)
                
                # Check if the input is in a compatible PCM16 format
                # Accept: pcm16, pcm16le (little-endian), pcm16be (big-endian)
                if chunk_encoding not in ["pcm16", "pcm16le", "pcm16be"]:
                    log.error(f"Invalid audio encoding: {chunk_encoding}. Expected pcm16.")
                    raise ValueError(f"Audio must be in PCM16 format, got: {chunk_encoding}")
                
                if chunk_sample_rate != sample_rate:
                    log.error(f"Invalid sample rate: {chunk_sample_rate}. Expected {sample_rate}.")
                    raise ValueError(f"Audio sample rate must be {sample_rate} Hz, got: {chunk_sample_rate} Hz")
                
                # Convert endianness if needed
                # ElevenLabs expects little-endian PCM16
                audio_to_send = audio_b64
                if chunk_encoding == "pcm16be":
                    # Convert big-endian to little-endian by swapping bytes
                    import struct
                    pcm_bytes = base64.b64decode(audio_b64)
                    # Swap bytes for each 16-bit sample
                    samples = struct.unpack(f'>{len(pcm_bytes)//2}h', pcm_bytes)
                    pcm_bytes_le = struct.pack(f'<{len(samples)}h', *samples)
                    audio_to_send = base64.b64encode(pcm_bytes_le).decode("utf-8")
                    log.debug("Converted big-endian to little-endian PCM16")
                
                # Audio is in correct format
                log.debug(f"Audio validated: {chunk_encoding} @ {sample_rate} Hz")

                # Build message
                message = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": audio_to_send,
                    "commit": False,  # Let VAD or manual strategy handle commits
                    "sample_rate": sample_rate,
                }

                # Only send previous_text with the first chunk if needed
                if first_chunk:
                    first_chunk = False
                    # Could add previous_text support here if needed

                log.debug(f"Sending audio chunk: {len(audio_to_send)} bytes (base64)")
                await websocket.send(json.dumps(message))

        log.debug("Producer loop completed")
        shared["done"] = True

        # Close the WebSocket gracefully
        log.debug("Closing WebSocket connection")
        await websocket.close()

    async def _consumer_loop(
        self,
        websocket,
        outputs: NodeOutputs,
        shared: dict,
    ) -> None:
        """Consume WebSocket events and stream transcription results.

        Args:
            websocket: WebSocket connection.
            outputs (NodeOutputs): Output emitter for streaming data.
            shared (dict): Shared state dict with flags/counters.
        """
        log.debug("Consumer loop started")

        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("message_type", "")
                log.debug(f"Received message type: {message_type}")

                # Handle session started
                if message_type == "session_started":
                    session_id = data.get("session_id", "")
                    log.info(f"Session started: {session_id}")
                    continue

                # Handle partial transcript
                elif message_type == "partial_transcript":
                    text = data.get("text", "")
                    if text:
                        log.debug(f"Partial transcript: {text}")
                        await outputs.emit(
                            "chunk",
                            Chunk(
                                content=text,
                                done=False,
                                content_type="text",
                            ),
                        )

                # Handle committed transcript
                elif message_type == "committed_transcript":
                    text = data.get("text", "")
                    if text:
                        log.debug(f"Committed transcript: {text}")
                        await outputs.emit(
                            "chunk",
                            Chunk(
                                content=text,
                                done=False,
                                content_type="text",
                            ),
                        )

                # Handle committed transcript with timestamps
                elif message_type == "committed_transcript_with_timestamps":
                    text = data.get("text", "")
                    language_code = data.get("language_code")
                    words = data.get("words", [])
                    
                    if text:
                        log.debug(f"Committed transcript with timestamps: {text}")
                        metadata = {}
                        if language_code:
                            metadata["language_code"] = language_code
                        if words:
                            metadata["words"] = words
                        
                        await outputs.emit(
                            "chunk",
                            Chunk(
                                content=text,
                                done=False,
                                content_type="text",
                                content_metadata=metadata,
                            ),
                        )

                # Handle errors
                elif "error" in message_type:
                    error_msg = data.get("error", "Unknown error")
                    log.error(f"Transcription error ({message_type}): {error_msg}")
                    raise RuntimeError(f"Transcription error: {error_msg}")

            except json.JSONDecodeError as e:
                log.error(f"Failed to decode WebSocket message: {e}")
                raise RuntimeError(f"Invalid WebSocket message: {e}")
            except Exception as e:
                log.error(f"Error processing WebSocket message: {e}")
                raise

        # Send final done signal
        log.debug("Sending final done chunk")
        await outputs.emit(
            "chunk",
            Chunk(content="", done=True, content_type="text"),
        )

        log.debug("Consumer loop completed")
