
import os
from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient

load_dotenv()

API_KEY = os.getenv("GNANI_API_KEY")

if not API_KEY:
    raise RuntimeError("GNANI_API_KEY is missing from .env")

client = GnaniSTTClient(api_key=API_KEY)


def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "recording.wav",
    language_code: str = "indic-auto",
) -> str:

    if not audio_bytes:
        raise ValueError("Audio recording is empty.")

    result = client.transcribe_bytes(
        audio_bytes,
        filename=filename,
        language_code=language_code,
        model="prisma-v3.0",
        format="transcribe",
    )

    transcript = result.get("transcript", "").strip()

    if not transcript:
        raise RuntimeError("Gnani returned an empty transcript.")

    return transcript
