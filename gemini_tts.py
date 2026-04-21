import base64
import io
import json
import re
import time
import wave

import requests

from tts_utils import split_text

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
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

GEMINI_CONNECT_TIMEOUT = 30
GEMINI_READ_TIMEOUT = 180

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


class ServerError(RuntimeError):
    """5xx가 재시도 후에도 계속 발생 — 분할-재시도 시도 대상."""
    pass


class NetworkError(RuntimeError):
    """연결 실패 / 타임아웃 / 스트림 중단 — 다른 키로 회전하면 통할 수 있음."""
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


def merge_wavs(wav_bytes_list: list) -> bytes:
    if not wav_bytes_list:
        return b""
    pcm_parts = []
    rate = 24000
    for wav in wav_bytes_list:
        pcm, rate = _wav_to_pcm(wav)
        pcm_parts.append(pcm)
    return _pcm_to_wav(b"".join(pcm_parts), rate=rate)


def synthesize_gemini(
    api_key: str,
    text: str,
    model: str,
    voice_name: str,
    style_prompt: str = "",
    seed: int = None,
    temperature: float = None,
    stats: dict = None,
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
    TRANSIENT_CODES = {500, 502, 503, 504}
    try:
        resp = requests.post(
            url,
            params={"key": api_key, "alt": "sse"},
            json=body,
            timeout=(GEMINI_CONNECT_TIMEOUT, GEMINI_READ_TIMEOUT),
            stream=True,
        )
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
        raise NetworkError(
            f"Gemini TTS 연결 지연 ({GEMINI_CONNECT_TIMEOUT}s). 잠시 후 다시 시도해 보세요."
        ) from e
    except requests.exceptions.ConnectionError as e:
        raise NetworkError(
            f"Gemini TTS 연결 실패: {e}"
        ) from e

    if resp.status_code != 200:
        body_text = resp.text
        if _is_quota_error(resp.status_code, body_text):
            raise QuotaExceeded(f"할당량 초과 ({resp.status_code}): {body_text}")
        if resp.status_code in TRANSIENT_CODES:
            raise ServerError(
                f"Gemini TTS 서버 오류 ({resp.status_code}). 구글 서버 문제입니다. "
                f"잠시 후 '이어서 재시도' 버튼으로 수동 재시도하세요.\n{body_text[:300]}"
            )
        raise RuntimeError(f"Gemini TTS API 오류 ({resp.status_code}): {body_text}")

    pcm_parts = []
    rate = 24000
    finish_reason = "UNKNOWN"
    last_error_msg = None
    start_time = time.monotonic()
    events_received = 0
    first_byte_at = None
    last_event_at = start_time

    if stats is not None:
        stats["events"] = 0
        stats["pcm_bytes"] = 0
        stats["ttfb"] = None
        stats["finish_reason"] = "UNKNOWN"
        stats["last_gap"] = 0.0

    try:
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            events_received += 1
            now = time.monotonic()
            if stats is not None:
                stats["events"] = events_received
                stats["last_gap"] = now - last_event_at
            last_event_at = now

            if "error" in event:
                err = event["error"]
                last_error_msg = err.get("message", json.dumps(err))
                code = err.get("code", 0)
                if _is_quota_error(code, last_error_msg):
                    raise QuotaExceeded(f"할당량 초과 ({code}): {last_error_msg}")
                if code in TRANSIENT_CODES:
                    raise ServerError(f"Gemini TTS 서버 오류 ({code}): {last_error_msg}")
                continue
            candidates = event.get("candidates", [])
            if not candidates:
                continue
            cand = candidates[0]
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData")
                if not inline or "data" not in inline:
                    continue
                decoded = base64.b64decode(inline["data"])
                pcm_parts.append(decoded)
                if first_byte_at is None:
                    first_byte_at = now - start_time
                    if stats is not None:
                        stats["ttfb"] = first_byte_at
                if stats is not None:
                    stats["pcm_bytes"] = sum(len(p) for p in pcm_parts)
                mime = inline.get("mimeType", "")
                m = re.search(r"rate=(\d+)", mime)
                if m:
                    rate = int(m.group(1))
            fr = cand.get("finishReason")
            if fr:
                finish_reason = fr
                if stats is not None:
                    stats["finish_reason"] = fr
    except (
        requests.exceptions.ReadTimeout,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ConnectionError,
    ) as e:
        total_elapsed = time.monotonic() - start_time
        total_bytes = sum(len(p) for p in pcm_parts)
        ttfb_str = f"{first_byte_at:.1f}s" if first_byte_at is not None else "없음"
        diag = (
            f"TTFB={ttfb_str} · 이벤트={events_received}회 · "
            f"수신={total_bytes:,}B · 경과={total_elapsed:.1f}s · "
            f"finishReason={finish_reason}"
        )
        if pcm_parts:
            raise NetworkError(
                f"Gemini TTS 스트림 중단 (부분 수신 후 {GEMINI_READ_TIMEOUT}s 무응답). "
                f"[{diag}] 다시 시도하세요.\n원인: {e}"
            ) from e
        raise NetworkError(
            f"Gemini TTS 응답 지연 (첫 바이트 대기 중 {GEMINI_READ_TIMEOUT}s 내 데이터 없음). "
            f"[{diag}] 서버 과부하 가능. 잠시 후 재시도.\n원인: {e}"
        ) from e
    finally:
        resp.close()

    if not pcm_parts:
        hints = {
            "OTHER": "출력 길이 초과 가능성 — 청크 크기를 줄여보세요",
            "MAX_TOKENS": "모델 출력 토큰 한도 초과 — 청크 크기 축소 필요",
            "SAFETY": "안전 필터에 의해 차단",
            "RECITATION": "저작권/인용 탐지에 의해 차단",
            "PROHIBITED_CONTENT": "금지된 콘텐츠",
            "BLOCKLIST": "차단 목록에 걸림",
        }
        hint = hints.get(finish_reason, last_error_msg or "원인 불명")
        msg = f"Gemini가 오디오를 반환하지 않음 (finishReason={finish_reason}): {hint}"
        if finish_reason in ("OTHER", "MAX_TOKENS"):
            raise OutputOverflow(msg)
        raise RuntimeError(msg)

    return _pcm_to_wav(b"".join(pcm_parts), rate=rate)


def _call_with_rotation(keys, fn, on_rotate=None):
    last_err = None
    for i, key in enumerate(keys):
        try:
            return fn(key), i
        except (QuotaExceeded, NetworkError) as e:
            last_err = e
            if on_rotate and i + 1 < len(keys):
                reason = "할당량 초과" if isinstance(e, QuotaExceeded) else "네트워크 오류"
                on_rotate(i, i + 1, f"{reason}: {e}")
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

    for i, chunk in enumerate(chunks):
        try:
            wavs, current_idx = synthesize_gemini_chunk(
                api_keys, current_idx, chunk, model, voice_name,
                style_prompt=style_prompt, seed=seed, temperature=temperature,
                rotate_cb=rotate_cb,
            )
        except QuotaExceeded as e:
            raise QuotaExceeded(
                f"모든 API 키의 할당량이 소진되었습니다 ({len(api_keys)}개 모두 실패). "
                f"백업 키를 추가하거나 내일까지 기다리세요. 마지막 응답: {e}"
            ) from e

        for wav in wavs:
            pcm, rate = _wav_to_pcm(wav)
            pcm_parts.append(pcm)
            wav_parts.append(wav)
        if progress_cb:
            progress_cb(i + 1, len(chunks), current_idx + 1)

    merged = _pcm_to_wav(b"".join(pcm_parts), rate=rate)
    return merged, wav_parts


def synthesize_gemini_chunk(
    api_keys,
    start_key_idx: int,
    chunk_text: str,
    model: str,
    voice_name: str,
    style_prompt: str = "",
    seed: int = None,
    temperature: float = None,
    rotate_cb=None,
    stats: dict = None,
    _depth: int = 0,
):
    """단일 청크를 처리. 키 로테이션 + OutputOverflow 자동 분할-재시도.
    Returns (list[wav_bytes], new_key_idx)."""
    if not api_keys:
        raise ValueError("API 키가 없습니다.")
    if start_key_idx >= len(api_keys) or start_key_idx < 0:
        start_key_idx = 0
    remaining_keys = api_keys[start_key_idx:]

    def _try(k):
        return synthesize_gemini(
            k, chunk_text, model, voice_name, style_prompt,
            seed=seed, temperature=temperature, stats=stats,
        )

    try:
        wav, offset = _call_with_rotation(remaining_keys, _try, on_rotate=rotate_cb)
        return [wav], start_key_idx + offset
    except OutputOverflow:
        if _depth >= 2 or len(chunk_text) <= 1:
            raise
        sub_chunks = _split_for_retry(chunk_text)
        if len(sub_chunks) < 2:
            raise
        all_wavs = []
        idx = start_key_idx
        for sub in sub_chunks:
            sub_wavs, idx = synthesize_gemini_chunk(
                api_keys, idx, sub, model, voice_name,
                style_prompt=style_prompt, seed=seed, temperature=temperature,
                rotate_cb=rotate_cb, stats=stats, _depth=_depth + 1,
            )
            all_wavs.extend(sub_wavs)
        return all_wavs, idx


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
