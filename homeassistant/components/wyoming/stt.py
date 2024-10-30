"""Support for Wyoming speech-to-text services."""

from collections.abc import AsyncIterable
import logging
import aiohttp
import asyncio

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient

from homeassistant.components import stt
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, SAMPLE_CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
from .data import WyomingService
from .error import WyomingError
from .models import DomainDataItem

_LOGGER = logging.getLogger(__name__)

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
API_KEY = "your_groq_api_key"  # Replace with your actual Groq API key
MODEL = "whisper-large-v3"  # Whisper model name

async def _transcribe_with_whisper_on_groq(self, metadata, stream):
    """Transcribe audio using Whisper on Groq."""

    # Convert async stream to a single audio buffer
    audio_data = bytearray()
    async for chunk in stream:
        audio_data.extend(chunk)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "audio/wav"
    }

    data = aiohttp.FormData()
    data.add_field("file", audio_data, filename="audio.wav", content_type="audio/wav")
    data.add_field("model", MODEL)
    data.add_field("language", metadata.language or "en")

    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, data=data) as response:
            if response.status == 200:
                result = await response.json()
                return stt.SpeechResult(result.get("text"), stt.SpeechResultState.SUCCESS)
            else:
                _LOGGER.error("Groq Whisper STT failed with status %d", response.status)
                return None  # Return None to indicate failure

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Wyoming speech-to-text."""
    item: DomainDataItem = hass.data[DOMAIN][config_entry.entry_id]
    async_add_entities(
        [
            WyomingSttProvider(config_entry, item.service),
        ]
    )

class WyomingSttProvider(stt.SpeechToTextEntity):
    """Wyoming speech-to-text provider with Whisper on Groq as primary."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        service: WyomingService,
    ) -> None:
        """Set up provider."""
        self.service = service
        asr_service = service.info.asr[0]

        model_languages: set[str] = set()
        for asr_model in asr_service.models:
            if asr_model.installed:
                model_languages.update(asr_model.languages)

        self._supported_languages = list(model_languages)
        self._attr_name = asr_service.name
        self._attr_unique_id = f"{config_entry.entry_id}-stt"

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return self._supported_languages

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return a list of supported formats."""
        return [stt.AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return a list of supported codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return a list of supported bitrates."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return a list of supported channels."""
        return [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process an audio stream using Whisper on Groq, with Wyoming as a backup."""

        # Try Whisper on Groq first
        whisper_result = await _transcribe_with_whisper_on_groq(self, metadata, stream)
        if whisper_result is not None:
            return whisper_result

        _LOGGER.info("Falling back to Wyoming STT")

        # Fall back to Wyoming STT if Whisper on Groq fails
        try:
            async with AsyncTcpClient(self.service.host, self.service.port) as client:
                await client.write_event(Transcribe(language=metadata.language).event())

                # Begin audio stream
                await client.write_event(
                    AudioStart(
                        rate=SAMPLE_RATE,
                        width=SAMPLE_WIDTH,
                        channels=SAMPLE_CHANNELS,
                    ).event(),
                )

                async for audio_bytes in stream:
                    chunk = AudioChunk(
                        rate=SAMPLE_RATE,
                        width=SAMPLE_WIDTH,
                        channels=SAMPLE_CHANNELS,
                        audio=audio_bytes,
                    )
                    await client.write_event(chunk.event())

                await client.write_event(AudioStop().event())

                while True:
                    event = await client.read_event()
                    if event is None:
                        _LOGGER.debug("Connection lost")
                        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

                    if Transcript.is_type(event.type):
                        transcript = Transcript.from_event(event)
                        text = transcript.text
                        break

        except (OSError, WyomingError):
            _LOGGER.exception("Error processing audio stream with Wyoming STT")
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        return stt.SpeechResult(
            text,
            stt.SpeechResultState.SUCCESS,
        )
