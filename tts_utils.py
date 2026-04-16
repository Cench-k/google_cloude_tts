import base64
import re
import requests

TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICES_ENDPOINT = "https://texttospeech.googleapis.com/v1/voices"
MAX_CHARS = 4500


def list_voices(api_key, language_code="ko-KR"):
    resp = requests.get(
        VOICES_ENDPOINT,
        params={"languageCode": language_code, "key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    voices = resp.json().get("voices", [])
    voices.sort(key=lambda v: v["name"])
    return voices


def split_text(text, max_chars=MAX_CHARS):
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?。！？\n])\s*", text)
    chunks = []
    current = ""
    for s in sentences:
        if not s:
            continue
        if len(current) + len(s) <= max_chars:
            current += s
        else:
            if current:
                chunks.append(current)
                current = ""
            if len(s) > max_chars:
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i : i + max_chars])
            else:
                current = s
    if current:
        chunks.append(current)
    return chunks


def synthesize(
    api_key,
    text,
    voice_name,
    language_code="ko-KR",
    speaking_rate=1.0,
    pitch=0.0,
):
    body = {
        "input": {"text": text},
        "voice": {"languageCode": language_code, "name": voice_name},
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": speaking_rate,
            "pitch": pitch,
        },
    }
    resp = requests.post(
        TTS_ENDPOINT,
        params={"key": api_key},
        json=body,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"TTS API 오류 ({resp.status_code}): {resp.text}")
    return base64.b64decode(resp.json()["audioContent"])


def synthesize_long(
    api_key,
    text,
    voice_name,
    language_code="ko-KR",
    speaking_rate=1.0,
    pitch=0.0,
    progress_cb=None,
):
    chunks = split_text(text)
    if not chunks:
        raise ValueError("입력 텍스트가 비어 있습니다.")
    parts = []
    for i, chunk in enumerate(chunks):
        audio = synthesize(
            api_key, chunk, voice_name, language_code, speaking_rate, pitch
        )
        parts.append(audio)
        if progress_cb:
            progress_cb(i + 1, len(chunks))
    return b"".join(parts), parts
