import aiohttp
import asyncio
import base64
import json
from pydantic import Field
from nodetool.metadata.types import AudioRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.providers import Chunk
from enum import Enum
from typing import TypedDict, ClassVar
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class VoiceIDEnum(str, Enum):
    """Available ElevenLabs voices"""

    ARIA = "Aria (American female, expressive)"
    ROGER = "Roger (American male, confident)"
    SARAH = "Sarah (American female, soft)"
    LAURA = "Laura (American female, upbeat)"
    CHARLIE = "Charlie (Australian male, natural)"
    GEORGE = "George (British male, warm)"
    CALLUM = "Callum (Transatlantic male, intense)"
    RIVER = "River (American non-binary, confident)"
    LIAM = "Liam (American male, articulate)"
    CHARLOTTE = "Charlotte (Swedish female, seductive)"
    ALICE = "Alice (British female, confident)"
    WILL = "Will (American male, friendly)"
    JESSICA = "Jessica (American female, expressive)"
    ERIC = "Eric (American male, friendly)"
    CHRIS = "Chris (American male, casual)"
    BRIAN = "Brian (American male, deep)"
    DANIEL = "Daniel (British male, authoritative)"
    LILY = "Lily (British female, warm)"
    BILL = "Bill (American male, trustworthy)"


VOICE_ID_MAPPING = {
    VoiceIDEnum.ARIA: "9BWtsMINqrJLrRacOk9x",
    VoiceIDEnum.ROGER: "CwhRBWXzGAHq8TQ4Fs17",
    VoiceIDEnum.SARAH: "EXAVITQu4vr4xnSDxMaL",
    VoiceIDEnum.LAURA: "FGY2WhTYpPnrIDTdsKH5",
    VoiceIDEnum.CHARLIE: "IKne3meq5aSn9XLyUdCD",
    VoiceIDEnum.GEORGE: "JBFqnCBsd6RMkjVDRZzb",
    VoiceIDEnum.CALLUM: "N2lVS1w4EtoT3dr4eOWO",
    VoiceIDEnum.RIVER: "SAz9YHcvj6GT2YYXdXww",
    VoiceIDEnum.LIAM: "TX3LPaxmHKxFdv7VOQHJ",
    VoiceIDEnum.CHARLOTTE: "XB0fDUnXU5powFXDhCwa",
    VoiceIDEnum.ALICE: "Xb7hH8MSUJpSbSDYk0k2",
    VoiceIDEnum.WILL: "bIHbv24MWmeRgasZH58o",
    VoiceIDEnum.JESSICA: "cgSgspJ2msm6clMCkdW9",
    VoiceIDEnum.ERIC: "cjVigY5qzO86Huf0OWal",
    VoiceIDEnum.CHRIS: "iP95p4xoKVk53GoZ742B",
    VoiceIDEnum.BRIAN: "nPczCjzI2devNBz1zQrb",
    VoiceIDEnum.DANIEL: "onwK4e9ZLuTAKqWW03F9",
    VoiceIDEnum.LILY: "pFZP5JQG7iQjIQuC4Bku",
    VoiceIDEnum.BILL: "pqHfZKP75CvOlQylNhV4",
}


class ModelID(str, Enum):
    """Available ElevenLabs models"""

    MULTILINGUAL_V2 = "eleven_multilingual_v2"  # Most life-like, 29 languages
    TURBO_V2_5 = "eleven_turbo_v2_5"  # High quality, low latency, 32 languages
    FLASH_V2_5 = "eleven_flash_v2_5"  # Ultra low latency, 32 languages
    TURBO_V2 = "eleven_turbo_v2"  # English-only, low latency
    FLASH_V2 = "eleven_flash_v2"  # Ultra low latency, English-only
    MULTILINGUAL_STS_V2 = "eleven_multilingual_sts_v2"  # Speech-to-speech, multilingual
    ENGLISH_STS_V2 = "eleven_english_sts_v2"  # Speech-to-speech, English
    MONOLINGUAL_V1 = "eleven_monolingual_v1"  # Legacy English model
    MULTILINGUAL_V1 = "eleven_multilingual_v1"  # Legacy multilingual model


class LanguageID(str, Enum):
    """Available languages for ElevenLabs models"""

    NONE = "none"
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE = "zh"
    GERMAN = "de"
    HINDI = "hi"
    FRENCH = "fr"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    SPANISH = "es"
    RUSSIAN = "ru"
    INDONESIAN = "id"
    DUTCH = "nl"
    TURKISH = "tr"
    FILIPINO = "fil"
    POLISH = "pl"
    SWEDISH = "sv"
    BULGARIAN = "bg"
    ROMANIAN = "ro"
    ARABIC = "ar"
    CZECH = "cs"
    GREEK = "el"
    FINNISH = "fi"
    CROATIAN = "hr"
    MALAY = "ms"
    SLOVAK = "sk"
    DANISH = "da"
    TAMIL = "ta"
    UKRAINIAN = "uk"
    VIETNAMESE = "vi"
    NORWEGIAN = "no"
    HUNGARIAN = "hu"


class TextNormalization(str, Enum):
    """Text normalization options"""

    AUTO = "auto"
    ON = "on"
    OFF = "off"


class OutputFormat(str, Enum):
    """Output format options for WebSocket streaming"""

    MP3_22050_32 = "mp3_22050_32"
    MP3_44100_32 = "mp3_44100_32"
    MP3_44100_64 = "mp3_44100_64"
    MP3_44100_96 = "mp3_44100_96"
    MP3_44100_128 = "mp3_44100_128"
    MP3_44100_192 = "mp3_44100_192"
    PCM_8000 = "pcm_8000"
    PCM_16000 = "pcm_16000"
    PCM_22050 = "pcm_22050"
    PCM_24000 = "pcm_24000"
    PCM_44100 = "pcm_44100"
    ULAW_8000 = "ulaw_8000"
    ALAW_8000 = "alaw_8000"
    OPUS_48000_32 = "opus_48000_32"
    OPUS_48000_64 = "opus_48000_64"
    OPUS_48000_96 = "opus_48000_96"
    OPUS_48000_128 = "opus_48000_128"
    OPUS_48000_192 = "opus_48000_192"


class TextToSpeech(BaseNode):
    """
    Generate natural-sounding speech using ElevenLabs' advanced text-to-speech technology. Features multiple voices and customizable parameters.
    audio, tts, speech, synthesis, voice

    Use cases:
    - Create professional voiceovers
    - Generate character voices
    - Produce multilingual content
    - Create audiobooks
    - Generate voice content
    """

    voice: VoiceIDEnum = Field(
        default=VoiceIDEnum.ARIA,
        description="Voice ID to be used for generation",
    )
    text: str = Field(
        default="Hello, how are you?",
        description="The text to convert to speech",
    )
    tts_model_id: ModelID = Field(
        default=ModelID.MONOLINGUAL_V1,
        description="The TTS model to use for generation",
    )
    voice_settings: dict = Field(
        default=None,
        description="Optional voice settings to override defaults",
    )
    language_code: LanguageID = Field(
        default=LanguageID.NONE,
        description="Language code to enforce (only works with Turbo v2.5)",
    )
    optimize_streaming_latency: int = Field(
        default=2,
        ge=0,
        le=4,
        description="Latency optimization level (0-4). Higher values trade quality for speed",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        le=4294967295,
        description="Seed for deterministic generation (0-4294967295). -1 means random",
    )
    text_normalization: TextNormalization = Field(
        default=TextNormalization.AUTO,
        description="Controls text normalization behavior",
    )
    stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice stability (0-1). Higher values make output more consistent, lower values more varied",
    )
    similarity_boost: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similarity to original voice (0-1). Higher values make output closer to original voice",
    )
    style: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Speaking style emphasis (0-1). Higher values increase style expression",
    )
    use_speaker_boost: bool = Field(
        default=False,
        description="Whether to use speaker boost for clearer, more consistent output",
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        api_key = context.environment.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is required")

        voice_id = VOICE_ID_MAPPING[self.voice]
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": self.text,
            "model_id": self.tts_model_id,
            "voice_settings": self.voice_settings,
            "optimize_streaming_latency": self.optimize_streaming_latency,
        }

        voice_settings = {}

        if self.language_code != LanguageID.NONE:
            payload["language_code"] = self.language_code
        if self.seed != -1:
            payload["seed"] = self.seed
        if self.text_normalization:
            payload["text_normalization"] = self.text_normalization
        if self.stability != 0.5:
            voice_settings["stability"] = self.stability
        if self.similarity_boost != 0.75:
            voice_settings["similarity_boost"] = self.similarity_boost
        if self.style != 0.0:
            voice_settings["style"] = self.style
        if self.use_speaker_boost:
            voice_settings["use_speaker_boost"] = self.use_speaker_boost

        if voice_settings:
            payload["voice_settings"] = voice_settings

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"ElevenLabs API error: {error_text}")

                audio_data = await response.read()
                return await context.audio_from_bytes(audio_data)

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["voice", "text"]


class RealtimeTextToSpeech(BaseNode):
    """
    Stream text-to-speech using ElevenLabs WebSocket API. Consumes text chunks and outputs audio chunks in real-time.
    audio, tts, speech, streaming, realtime, websocket

    Use cases:
    - Real-time voice generation from streaming text
    - Interactive voice applications
    - Low-latency text-to-speech conversion
    - Streaming dialogue generation
    """

    voice: VoiceIDEnum = Field(
        default=VoiceIDEnum.ARIA,
        description="Voice ID to be used for generation",
    )
    chunk: Chunk = Field(
        title="Chunk",
        default=Chunk(),
        description="The text chunk to use as input.",
    )
    model_id: ModelID = Field(
        default=ModelID.TURBO_V2_5,
        description="The TTS model to use for generation",
    )
    language_code: LanguageID = Field(
        default=LanguageID.NONE,
        description="Language code to enforce (only works with Turbo v2.5)",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP3_44100_128,
        description="Audio output format for streaming",
    )
    stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice stability (0-1). Higher values make output more consistent",
    )
    similarity_boost: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similarity to original voice (0-1)",
    )
    style: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Speaking style emphasis (0-1)",
    )
    use_speaker_boost: bool = Field(
        default=True,
        description="Whether to use speaker boost for clearer output",
    )
    speed: float = Field(
        default=1.0,
        ge=0.7,
        le=1.2,
        description="Speed of the generated speech (0.7-1.2)",
    )
    enable_ssml_parsing: bool = Field(
        default=False,
        description="Enable SSML parsing in text input",
    )

    _supports_dynamic_outputs: ClassVar[bool] = False

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        chunk: Chunk

    @classmethod
    def return_type(cls):
        return cls.OutputType

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["voice", "chunk", "model_id"]

    async def _producer_loop(
        self,
        websocket,
        inputs: NodeInputs,
        shared: dict,
    ) -> None:
        """Continuously read streaming text inputs and forward them to the WebSocket.

        Args:
            websocket: WebSocket connection.
            inputs (NodeInputs): Streaming workflow inputs.
            shared (dict): Shared state dict with flags/counters.
        """
        log.debug("Producer loop started")

        async for handle, item in inputs.any():
            if handle == "chunk":
                assert isinstance(item, Chunk)
                
                # Send text chunks immediately as they arrive
                if item.content and not item.done:
                    message = {"text": item.content + " "}
                    log.debug(f"Sending text chunk: {len(item.content)} characters")
                    await websocket.send(json.dumps(message))
                
                # When done, just log (we'll close the stream after the loop)
                if item.done:
                    log.debug("Received done signal")
            else:
                log.error(f"Unknown handle in producer loop: {handle}")
                raise ValueError(f"Unknown handle: {handle}")

        # Send empty string to close the stream
        log.debug("Producer loop completed, sending close message")
        await websocket.send(json.dumps({"text": ""}))
        shared["done_producing"] = True

    async def _consumer_loop(
        self,
        websocket,
        outputs: NodeOutputs,
        shared: dict,
    ) -> None:
        """Consume WebSocket events and stream audio chunks.

        Args:
            websocket: WebSocket connection.
            outputs (NodeOutputs): Output emitter for streaming data.
            shared (dict): Shared state dict with flags/counters.
        """
        log.debug("Consumer loop started")

        async for message in websocket:
            try:
                data = json.loads(message)
                log.debug(f"Received WebSocket message: {list(data.keys())}")

                # Check for final message
                if data.get("isFinal"):
                    log.info("Received final message, ending stream")
                    await outputs.emit(
                        "chunk", Chunk(content="", done=True, content_type="audio")
                    )
                    break

                # Extract audio chunk
                if "audio" in data:
                    audio_b64 = data["audio"]
                    if audio_b64:
                        log.debug(f"Received audio chunk: {len(audio_b64)} bytes (base64)")
                        await outputs.emit(
                            "chunk",
                            Chunk(
                                content=audio_b64,
                                done=False,
                                content_type="audio",
                            ),
                        )

            except json.JSONDecodeError as e:
                log.error(f"Failed to decode WebSocket message: {e}")
                raise RuntimeError(f"Invalid WebSocket message: {e}")
            except Exception as e:
                log.error(f"Error processing WebSocket message: {e}")
                raise

        log.debug("Consumer loop completed")

    async def run(
        self,
        context: ProcessingContext,
        inputs: NodeInputs,
        outputs: NodeOutputs,
    ) -> None:
        """Run the realtime TTS with streaming input/output.

        Args:
            context (ProcessingContext): Workflow execution context.
            inputs (NodeInputs): Streaming text inputs.
            outputs (NodeOutputs): Output emitter for streaming audio chunks.
        """
        import websockets

        api_key = context.environment.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is required")

        voice_id = VOICE_ID_MAPPING[self.voice]

        # Build WebSocket URL with query parameters
        base_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        params = {
            "model_id": self.model_id.value,
            "output_format": self.output_format.value,
            "enable_ssml_parsing": str(self.enable_ssml_parsing).lower(),
        }
        if self.language_code != LanguageID.NONE:
            params["language_code"] = self.language_code.value

        # Build query string
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        ws_url = f"{base_url}?{query_string}"

        log.info(f"Connecting to ElevenLabs WebSocket: voice={self.voice.value}")

        # Connect to WebSocket with API key in headers
        headers = {"xi-api-key": api_key}

        try:
            async with websockets.connect(ws_url, extra_headers=headers) as websocket:
                log.info("Connected to ElevenLabs WebSocket")

                # Send initialization message with voice settings
                voice_settings = {
                    "stability": self.stability,
                    "similarity_boost": self.similarity_boost,
                    "style": self.style,
                    "use_speaker_boost": self.use_speaker_boost,
                    "speed": self.speed,
                }

                init_message = {
                    "text": " ",  # Must be a single space for initialization
                    "voice_settings": voice_settings,
                }

                log.debug("Sending initialization message")
                await websocket.send(json.dumps(init_message))

                # Run producer and consumer loops concurrently
                shared = {"done_producing": False}
                log.info("Starting producer and consumer loops")
                await asyncio.gather(
                    self._producer_loop(websocket, inputs, shared),
                    self._consumer_loop(websocket, outputs, shared),
                )
                log.info("RealtimeTextToSpeech execution completed")

        except websockets.exceptions.WebSocketException as e:
            log.error(f"WebSocket error: {e}")
            raise RuntimeError(f"WebSocket connection failed: {e}")
        except Exception as e:
            log.error(f"Error in realtime TTS: {e}")
            raise
