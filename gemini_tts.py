import base64
import io
import re
import wave

import requests

from tts_utils import split_text

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

GEMINI_MODELS = [
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]

GEMINI_VOICES = [
    ("Zephyr", "Bright 밝음"),
    ("Puck", "Upbeat 경쾌"),
    ("Charon", "Informative 정보전달"),
    ("Kore", "Firm 단호"),
    ("Fenrir", "Excitable 흥분"),
    ("Leda", "Youthful 젊음"),
    ("Orus", "Firm 단호"),
    ("Aoede", "Breezy 산뜻"),
    ("Callirrhoe", "Easy-going 편안"),
    ("Autonoe", "Bright 밝음"),
    ("Enceladus", "Breathy 숨소리"),
    ("Iapetus", "Clear 또렷"),
    ("Umbriel", "Easy-going 편안"),
    ("Algieba", "Smooth 부드러움"),
    ("Despina", "Smooth 부드러움"),
    ("Erinome", "Clear 또렷"),
    ("Algenib", "Gravelly 거친톤"),
    ("Rasalgethi", "Informative 정보전달"),
    ("Laomedeia", "Upbeat 경쾌"),
    ("Achernar", "Soft 부드러움"),
    ("Alnilam", "Firm 단호"),
    ("Schedar", "Even 균일"),
    ("Gacrux", "Mature 성숙"),
    ("Pulcherrima", "Forward 직설"),
    ("Achird", "Friendly 친근"),
    ("Zubenelgenubi", "Casual 캐주얼"),
    ("Vindemiatrix", "Gentle 온화"),
    ("Sadachbia", "Lively 활기"),
    ("Sadaltager", "Knowledgeable 지적"),
    ("Sulafat", "Warm 따뜻"),
]

GEMINI_MAX_BYTES = 4000


def _pcm_to_wav(pcm_bytes: bytes, rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(rate)
        wav.writeframes(pcm_bytes)
    return buf.getvalue()


def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        return wav.readframes(wav.getnframes()), wav.getframerate()


def synthesize_gemini(
    api_key: str,
    text: str,
    model: str,
    voice_name: str,
    style_prompt: str = "",
) -> bytes:
    content_text = f"Say {style_prompt}: {text}" if style_prompt.strip() else text
    body = {
        "contents": [{"parts": [{"text": content_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice_name},
                },
            },
        },
    }
    url = GEMINI_ENDPOINT.format(model=model)
    resp = requests.post(
        url,
        params={"key": api_key},
        json=body,
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini TTS API 오류 ({resp.status_code}): {resp.text}")
    data = resp.json()
    try:
        inline = data["candidates"][0]["content"]["parts"][0]["inlineData"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini 응답 파싱 실패: {data}")
    pcm = base64.b64decode(inline["data"])
    mime = inline.get("mimeType", "")
    m = re.search(r"rate=(\d+)", mime)
    rate = int(m.group(1)) if m else 24000
    return _pcm_to_wav(pcm, rate=rate)


def synthesize_gemini_long(
    api_key: str,
    text: str,
    model: str,
    voice_name: str,
    style_prompt: str = "",
    progress_cb=None,
):
    chunks = split_text(text, max_bytes=GEMINI_MAX_BYTES)
    if not chunks:
        raise ValueError("입력 텍스트가 비어 있습니다.")
    pcm_parts = []
    wav_parts = []
    rate = 24000
    for i, chunk in enumerate(chunks):
        wav = synthesize_gemini(api_key, chunk, model, voice_name, style_prompt)
        pcm, rate = _wav_to_pcm(wav)
        pcm_parts.append(pcm)
        wav_parts.append(wav)
        if progress_cb:
            progress_cb(i + 1, len(chunks))
    merged = _pcm_to_wav(b"".join(pcm_parts), rate=rate)
    return merged, wav_parts
