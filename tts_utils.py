import base64
import re
import requests

TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICES_ENDPOINT = "https://texttospeech.googleapis.com/v1/voices"
MAX_BYTES = 4500


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


def _bytelen(s: str) -> int:
    return len(s.encode("utf-8"))


def _force_split_by_bytes(s: str, max_bytes: int) -> list[str]:
    out = []
    buf = ""
    for ch in s:
        if _bytelen(buf) + _bytelen(ch) > max_bytes:
            out.append(buf)
            buf = ch
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def _refine_long_sentence(s: str, max_sentence_bytes: int) -> list[str]:
    if _bytelen(s) <= max_sentence_bytes:
        return [s]
    parts = re.split(r"(?<=[,，、;；:])\s*", s)
    out = []
    for p in parts:
        if _bytelen(p) <= max_sentence_bytes:
            out.append(p)
            continue
        buf = ""
        for word in re.split(r"(\s+)", p):
            if not word:
                continue
            if _bytelen(buf) + _bytelen(word) > max_sentence_bytes:
                if buf:
                    out.append(buf)
                    buf = ""
                if _bytelen(word) > max_sentence_bytes:
                    out.extend(_force_split_by_bytes(word, max_sentence_bytes))
                else:
                    buf = word
            else:
                buf += word
        if buf:
            out.append(buf)
    return out


def _split_paragraph_by_sentences(para, max_bytes, max_sentence_bytes=None):
    sentences = re.split(r"(?<=[.!?。！？\n])\s*", para)
    units = []
    for s in sentences:
        if not s:
            continue
        if max_sentence_bytes and _bytelen(s) > max_sentence_bytes:
            for piece in _refine_long_sentence(s, max_sentence_bytes):
                units.append((piece, True))
        else:
            units.append((s, False))

    chunks = []
    current = ""
    for unit, standalone in units:
        if standalone:
            if current:
                chunks.append(current)
                current = ""
            if _bytelen(unit) > max_bytes:
                chunks.extend(_force_split_by_bytes(unit, max_bytes))
            else:
                chunks.append(unit)
            continue
        if _bytelen(current) + _bytelen(unit) <= max_bytes:
            current += unit
        else:
            if current:
                chunks.append(current)
                current = ""
            if _bytelen(unit) > max_bytes:
                pieces = _force_split_by_bytes(unit, max_bytes)
                chunks.extend(pieces[:-1])
                current = pieces[-1] if pieces else ""
            else:
                current = unit
    if current:
        chunks.append(current)
    return chunks


def split_text(text, max_bytes=MAX_BYTES, max_sentence_bytes=None):
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current = ""
    sep = "\n\n"

    for para in paragraphs:
        if current and _bytelen(current) + _bytelen(sep) + _bytelen(para) <= max_bytes:
            current += sep + para
            continue

        if current:
            chunks.append(current)
            current = ""

        if _bytelen(para) <= max_bytes:
            current = para
            continue

        sub_chunks = _split_paragraph_by_sentences(
            para, max_bytes, max_sentence_bytes=max_sentence_bytes
        )
        if not sub_chunks:
            continue
        chunks.extend(sub_chunks[:-1])
        current = sub_chunks[-1]

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
    max_sentence_bytes=None,
    progress_cb=None,
):
    chunks = split_text(text, max_sentence_bytes=max_sentence_bytes)
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
