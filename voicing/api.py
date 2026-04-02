# voice/api.py
"""
Голосовой модуль для Tanym Loop.
Поддерживает Speech-to-Text через Faster Whisper (офлайн).
"""

import os
import tempfile
import wave
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from faster_whisper import WhisperModel

router = APIRouter(prefix="/voice", tags=["voice"])

# Инициализация модели Whisper
# Используем smaller model для быстрой работы, можно заменить на "large-v3"
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

print(f"🎤 Инициализация Whisper модели: {WHISPER_MODEL_SIZE}")

try:
    whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
	download_root=os.getenv("HF_HOME", "app/models_cache") + "/whisper"
    )
    print("✅ Whisper модель загружена")
except Exception as e:
    print(f"⚠️  Whisper модель не загружена: {e}")
    whisper_model = None


def transcribe_audio(audio_path: str) -> tuple[str, str, float]:
    """
    Распознаёт аудио файл через Faster Whisper.

    Returns:
        (text, language, confidence)
    """
    if not whisper_model:
        raise HTTPException(
            status_code=503,
            detail="Whisper model not initialized. Install faster-whisper."
        )

    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            language="ru",  # Можно сделать auto-detect
            vad_filter=True  # Фильтрация пауз
        )

        text = " ".join([segment.text for segment in segments])

        return text.strip(), info.language, info.language_probability

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")


@router.post("/stt")
async def speech_to_text(
        audio: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)")
):
    """
    Speech-to-Text endpoint.
    Принимает аудио файл и возвращает распознанный текст.

    **Поддерживаемые форматы:** WAV, MP3, M4A, FLAC
    """

    if not whisper_model:
        raise HTTPException(
            status_code=503,
            detail="Speech recognition unavailable. Install faster-whisper: pip install faster-whisper"
        )

    # Проверка расширения файла
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = os.path.splitext(audio.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}. Allowed: {allowed_extensions}"
        )

    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Распознаём
        text, lang, confidence = transcribe_audio(tmp_path)

        if not text:
            return {
                "ok": False,
                "text": "",
                "message": "Не удалось распознать речь. Попробуйте говорить громче."
            }

        return {
            "ok": True,
            "text": text,
            "language": lang,
            "confidence": round(confidence, 2),
            "message": "Распознавание успешно"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Удаляем временный файл
        try:
            os.unlink(tmp_path)
        except:
            pass


@router.get("/health")
def voice_health():
    """
    Проверка доступности голосового модуля.
    """
    return {
        "whisper_available": whisper_model is not None,
        "model": WHISPER_MODEL_SIZE if whisper_model else None,
        "device": WHISPER_DEVICE if whisper_model else None
    }
