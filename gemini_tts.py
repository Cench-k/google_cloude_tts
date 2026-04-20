import base64
import io
import re
import time
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

GEMINI_TIMEOUT = 300

MODEL_MAX_BYTES = {
    "gemini-3.1-flash-tts-preview": 1500,
    "gemini-2.5-flash-preview-tts": 1200,
    "gemini-2.5-pro-preview-tts": 1200,
}
DEFAULT_MAX_BYTES = 1200


def get_max_bytes(model: str) -> int:
    return MODEL_MAX_BYTES.get(model, DEFAULT_MAX_BYTES)


class QuotaExceeded(RuntimeError):
    pass


class OutputOverflow(RuntimeError):
    """finishReason=OTHER/MAX_TOKENS — 청크를 더 작게 쪼개서 재시도 가능."""
    pass


def _is_quota_error(status_code: int, body_text: str) -> bool:
    if status_code == 429:
        return True
    if status_code in (403, 400):
        lower = body_text.lower()
        if "quota" in lower or "resource_exhausted" in lower or "rate limit" in lower:
            return True
    return False


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
    seed: int = None,
    temperature: float = None,
) -> bytes:
    content_text = f"Say {style_prompt}: {text}" if style_prompt.strip() else text
    generation_config = {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": voice_name},
            },
        },
    }
    if seed is not None:
        generation_config["seed"] = int(seed)
    if temperature is not None:
        generation_config["temperature"] = float(temperature)
    body = {
        "contents": [{"parts": [{"text": content_text}]}],
        "generationConfig": generation_config,
    }
    url = GEMINI_ENDPOINT.format(model=model)
    MAX_ATTEMPTS = 4
    TRANSIENT_CODES = {500, 502, 503, 504}
    last_error = None
    resp = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.post(
                url,
                params={"key": api_key},
                json=body,
                timeout=GEMINI_TIMEOUT,
            )
        except requests.exceptions.ReadTimeout as e:
            last_error = e
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(
                f"Gemini TTS 응답 지연 (타임아웃 {GEMINI_TIMEOUT}s × {MAX_ATTEMPTS}회 재시도 실패). "
                f"청크 크기를 더 줄이거나 잠시 후 다시 시도해 보세요."
            ) from e

        if resp.status_code in TRANSIENT_CODES and attempt < MAX_ATTEMPTS - 1:
            last_error = RuntimeError(f"{resp.status_code}: {resp.text[:200]}")
            time.sleep(2 * (attempt + 1))
            continue
        break

    if resp is None:
        raise RuntimeError(f"Gemini TTS 호출 실패: {last_error}")
    if resp.status_code != 200:
        if _is_quota_error(resp.status_code, resp.text):
            raise QuotaExceeded(f"할당량 초과 ({resp.status_code}): {resp.text}")
        if resp.status_code in TRANSIENT_CODES:
            raise RuntimeError(
                f"Gemini TTS 서버 부하 지속 ({resp.status_code}, {MAX_ATTEMPTS}회 재시도 후에도 실패). "
                f"잠시 후 다시 시도해 보세요.\n{resp.text[:300]}"
            )
        raise RuntimeError(f"Gemini TTS API 오류 ({resp.status_code}): {resp.text}")
    data = resp.json()
    try:
        candidate = data["candidates"][0]
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini 응답에 candidate 없음: {data}")
    parts = candidate.get("content", {}).get("parts", [])
    finish_reason = candidate.get("finishReason", "UNKNOWN")
    if not parts:
        hints = {
            "OTHER": "출력 길이 초과 가능성 — 청크 크기를 줄여보세요",
            "MAX_TOKENS": "모델 출력 토큰 한도 초과 — 청크 크기 축소 필요",
            "SAFETY": "안전 필터에 의해 차단",
            "RECITATION": "저작권/인용 탐지에 의해 차단",
            "PROHIBITED_CONTENT": "금지된 콘텐츠",
            "BLOCKLIST": "차단 목록에 걸림",
        }
        hint = hints.get(finish_reason, "원인 불명")
        msg = f"Gemini가 오디오를 반환하지 않음 (finishReason={finish_reason}): {hint}"
        if finish_reason in ("OTHER", "MAX_TOKENS"):
            raise OutputOverflow(msg)
        raise RuntimeError(msg)
    try:
        inline = parts[0]["inlineData"]
    except (KeyError, TypeError):
        raise RuntimeError(f"응답 parts에 inlineData 없음: {parts[0]}")
    pcm = base64.b64decode(inline["data"])
    mime = inline.get("mimeType", "")
    m = re.search(r"rate=(\d+)", mime)
    rate = int(m.group(1)) if m else 24000
    return _pcm_to_wav(pcm, rate=rate)


def _call_with_rotation(keys, fn, on_rotate=None):
    last_err = None
    for i, key in enumerate(keys):
        try:
            return fn(key), i
        except QuotaExceeded as e:
            last_err = e
            if on_rotate and i + 1 < len(keys):
                on_rotate(i, i + 1, str(e))
            continue
    raise last_err or RuntimeError("사용 가능한 API 키가 없습니다.")


def synthesize_gemini_long(
    api_keys,
    text: str,
    model: str,
    voice_name: str,
    style_prompt: str = "",
    seed: int = None,
    temperature: float = None,
    progress_cb=None,
    rotate_cb=None,
):
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    api_keys = [k for k in api_keys if k]
    if not api_keys:
        raise ValueError("API 키가 없습니다.")

    chunks = split_text(text, max_bytes=get_max_bytes(model))
    if not chunks:
        raise ValueError("입력 텍스트가 비어 있습니다.")

    pcm_parts = []
    wav_parts = []
    rate = 24000
    current_idx = 0

    def _synthesize_one(chunk_text, depth=0):
        nonlocal current_idx
        remaining_keys = api_keys[current_idx:]

        def _try(k, c=chunk_text):
            return synthesize_gemini(
                k, c, model, voice_name, style_prompt,
                seed=seed, temperature=temperature,
            )

        try:
            wav, offset = _call_with_rotation(remaining_keys, _try, on_rotate=rotate_cb)
        except OutputOverflow:
            if depth >= 4 or len(chunk_text) <= 1:
                raise
            sub_chunks = _split_for_retry(chunk_text)
            if len(sub_chunks) < 2:
                raise
            wavs = []
            for sub in sub_chunks:
                sub_wav = _synthesize_one(sub, depth=depth + 1)
                wavs.append(sub_wav)
            return wavs
        current_idx += offset
        return wav

    def _flatten(wavs_or_wav):
        if isinstance(wavs_or_wav, list):
            out = []
            for w in wavs_or_wav:
                out.extend(_flatten(w))
            return out
        return [wavs_or_wav]

    for i, chunk in enumerate(chunks):
        try:
            result = _synthesize_one(chunk)
        except QuotaExceeded as e:
            raise QuotaExceeded(
                f"모든 API 키의 할당량이 소진되었습니다 ({len(api_keys)}개 모두 실패). "
                f"백업 키를 추가하거나 내일까지 기다리세요. 마지막 응답: {e}"
            ) from e

        for wav in _flatten(result):
            pcm, rate = _wav_to_pcm(wav)
            pcm_parts.append(pcm)
            wav_parts.append(wav)
        if progress_cb:
            progress_cb(i + 1, len(chunks), current_idx + 1)

    merged = _pcm_to_wav(b"".join(pcm_parts), rate=rate)
    return merged, wav_parts


def _split_for_retry(text: str) -> list:
    """OutputOverflow 발생 청크를 문단/문장/반 기준으로 2개 이상으로 쪼갠다."""
    if "\n\n" in text:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(parts) >= 2:
            mid = len(parts) // 2
            return ["\n\n".join(parts[:mid]), "\n\n".join(parts[mid:])]
    sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
    if len(sentences) >= 2:
        mid = len(sentences) // 2
        return [" ".join(sentences[:mid]).strip(), " ".join(sentences[mid:]).strip()]
    mid = len(text) // 2
    return [text[:mid].strip(), text[mid:].strip()]
