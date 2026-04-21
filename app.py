import io
import time
import zipfile
from datetime import datetime

import streamlit as st

from file_utils import read_file
from gemini_tts import (
    GEMINI_MODELS,
    GEMINI_VOICES,
    QuotaExceeded,
    get_max_bytes,
    merge_wavs,
    synthesize_gemini,
    synthesize_gemini_chunk,
)
from tts_utils import list_voices, split_text, synthesize

st.set_page_config(page_title="Google TTS 도우미", page_icon="🔊", layout="wide")
st.title("🔊 Google TTS 도우미")


SAMPLE_TEXTS = {
    "ko-KR": "안녕하세요. 오늘도 좋은 하루 되세요.",
    "en-US": "Hello. Have a wonderful day.",
    "ja-JP": "こんにちは。良い一日をお過ごしください。",
    "zh-CN": "你好。祝你今天愉快。",
}


def _voice_tier(voice_name: str) -> str:
    parts = voice_name.split("-")
    if len(parts) < 3:
        return "Unknown"
    tier = parts[2]
    if tier == "Chirp3" and len(parts) >= 4 and parts[3] == "HD":
        return "Chirp3-HD"
    return tier


@st.cache_data(ttl=3600, show_spinner="목소리 목록 로딩 중...")
def get_cloud_voices(key, lang):
    return list_voices(key, lang)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cloud_sample(key, voice, lang):
    return synthesize(
        key, SAMPLE_TEXTS.get(lang, SAMPLE_TEXTS["en-US"]),
        voice, language_code=lang,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_gemini_sample(key, model, voice):
    return synthesize_gemini(
        key, "안녕하세요. 오늘도 좋은 하루 되세요.", model, voice,
    )


with st.sidebar:
    st.header("⚙️ 설정")
    engine = st.radio(
        "엔진",
        ["Gemini TTS (신규)", "Cloud TTS"],
        index=0,
        help="Gemini TTS는 Zephyr/Puck 등 캐릭터 목소리와 스타일 프롬프트 지원",
    )

def _parse_keys(raw: str) -> list:
    if not raw:
        return []
    parts = []
    for line in raw.replace(",", "\n").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            parts.append(line)
    return parts


with st.sidebar:
    with st.expander("🔑 API 키 입력", expanded=True):
        if engine == "Gemini TTS (신규)":
            gemini_keys_raw = st.text_area(
                "Gemini API 키 (한 줄에 하나)",
                key="gemini_keys_raw",
                height=120,
                placeholder="AIzaSy...\nAIzaSy... (여러 개면 할당량 초과 시 자동 전환)",
                help="AI Studio(https://aistudio.google.com/apikey)에서 발급. "
                     "여러 키 입력 시 위에서부터 순서대로 로테이션.",
            )
            gemini_key_pool = _parse_keys(gemini_keys_raw)
        else:
            cloud_key_input = st.text_input(
                "Google Cloud TTS API 키",
                key="cloud_key_input",
                type="password",
                placeholder="AIzaSy...",
                help="Google Cloud 콘솔 → APIs & Services → Credentials",
            )
            cloud_key = cloud_key_input.strip() if cloud_key_input else ""

if engine == "Gemini TTS (신규)":
    if not gemini_key_pool:
        with st.sidebar:
            st.warning("Gemini API 키를 입력하세요.")
        st.info("👈 사이드바의 '🔑 API 키 입력'에 Gemini API 키를 입력해주세요.")
        st.stop()
    active_key = gemini_key_pool[0]
else:
    if not cloud_key:
        with st.sidebar:
            st.warning("Google Cloud TTS 키를 입력하세요.")
        st.info("👈 사이드바의 '🔑 API 키 입력'에 Cloud TTS API 키를 입력해주세요.")
        st.stop()
    active_key = cloud_key

with st.sidebar:
    if engine == "Gemini TTS (신규)":
        model_options = GEMINI_MODELS + ["(사용자 지정)"]
        model_choice = st.selectbox("모델", model_options, index=0)
        if model_choice == "(사용자 지정)":
            model = st.text_input(
                "모델 ID",
                value="gemini-2.5-flash-preview-tts",
                help="AI Studio에서 확인한 정확한 모델 ID 입력",
            )
        else:
            model = model_choice

        voice_labels = [f"{name} · {desc}" for name, desc in GEMINI_VOICES]
        voice_idx = st.selectbox(
            "목소리",
            range(len(voice_labels)),
            format_func=lambda i: voice_labels[i],
        )
        voice_name = GEMINI_VOICES[voice_idx][0]

        if st.button("🎧 이 목소리 샘플 듣기", use_container_width=True):
            with st.spinner("샘플 생성 중..."):
                last_err = None
                sample_audio = None
                for idx, key in enumerate(gemini_key_pool):
                    try:
                        sample_audio = get_gemini_sample(key, model, voice_name)
                        if idx > 0:
                            st.caption(f"키 #{idx + 1}로 샘플 생성됨 (앞 키들은 할당량 초과)")
                        break
                    except QuotaExceeded as e:
                        last_err = e
                        continue
                    except Exception as e:
                        last_err = e
                        break
                if sample_audio is not None:
                    st.audio(sample_audio, format="audio/wav")
                else:
                    st.error(f"샘플 생성 실패: {last_err}")

        style_prompt = st.text_input(
            "스타일 프롬프트 (선택)",
            placeholder="예: in a cheerful tone / 차분하게 / 뉴스 앵커처럼",
            help="'Say {프롬프트}: {본문}' 형식으로 전달되어 톤/감정 제어",
        )

        with st.expander("🎯 목소리 일관성 (청크 간 드리프트 방지)"):
            consistency_mode = st.checkbox(
                "일관성 모드",
                value=True,
                help="seed 고정 + 낮은 temperature로 청크 간 톤 유지",
            )
            if consistency_mode:
                seed_value = st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=2**31 - 1,
                    value=42,
                    step=1,
                    help="같은 seed는 같은 목소리 특성을 재현. 결과가 이상하면 값 변경",
                )
                temperature_value = st.slider(
                    "Temperature",
                    0.0, 2.0, 0.5, 0.05,
                    help="낮을수록 일관적, 높을수록 다양함 (0.3~0.7 권장)",
                )
            else:
                seed_value = None
                temperature_value = None

        st.caption(f"🔑 총 {len(gemini_key_pool)}개 키 준비됨 · 청크 크기: {get_max_bytes(model)}B")

        language_code = None
        speaking_rate = None
        pitch = None
        max_sentence_bytes = None
        is_chirp = False
        st.caption("💡 Gemini TTS는 언어 자동 감지, 30개 캐릭터 목소리 제공")
    else:
        language_code = st.selectbox(
            "언어",
            ["ko-KR", "en-US", "ja-JP", "zh-CN"],
            index=0,
        )

        try:
            voices = get_cloud_voices(active_key, language_code)
        except Exception as e:
            st.error(f"목소리 목록 로드 실패: {e}")
            st.stop()

        if not voices:
            st.error("사용 가능한 목소리가 없습니다.")
            st.stop()

        tiers = sorted({_voice_tier(v["name"]) for v in voices})
        tier_filter = st.multiselect(
            "목소리 등급",
            tiers,
            default=tiers,
        )

        filtered = (
            [v for v in voices if _voice_tier(v["name"]) in tier_filter]
            if tier_filter
            else voices
        )
        if not filtered:
            st.warning("필터에 맞는 목소리가 없습니다.")
            filtered = voices

        voice_options = {
            f"[{_voice_tier(v['name'])}] {v['name']} · {v['ssmlGender']}": v["name"]
            for v in filtered
        }
        voice_label = st.selectbox("목소리 선택", list(voice_options.keys()))
        voice_name = voice_options[voice_label]

        if st.button("🎧 이 목소리 샘플 듣기", use_container_width=True):
            with st.spinner("샘플 생성 중..."):
                try:
                    sample_audio = get_cloud_sample(active_key, voice_name, language_code)
                    st.audio(sample_audio, format="audio/mp3")
                except Exception as e:
                    st.error(f"샘플 생성 실패: {e}")

        is_chirp = "Chirp3" in voice_name
        speaking_rate = st.slider(
            "속도",
            0.25, 2.0 if is_chirp else 4.0,
            1.0, 0.05,
        )
        pitch = st.slider(
            "피치", -20.0, 20.0, 0.0, 1.0,
            disabled=is_chirp,
            help="Chirp3-HD는 피치 조절 미지원" if is_chirp else None,
        )
        if is_chirp:
            pitch = 0.0
        max_sentence_bytes = 300 if is_chirp else None
        model = None
        style_prompt = ""
        st.caption("💡 Chirp3-HD > Neural2 > Wavenet > Standard 순으로 자연스러움")

tab1, tab2 = st.tabs(["✏️ 직접 입력", "📁 파일 업로드"])

with tab1:
    text_direct = st.text_area(
        "텍스트 입력",
        height=300,
        placeholder="여기에 텍스트를 입력하세요...",
        key="direct_text",
    )

with tab2:
    uploaded = st.file_uploader("파일 선택", type=["txt", "docx", "pdf"])
    text_file = ""
    if uploaded:
        try:
            text_file = read_file(uploaded)
            st.success(f"파일 로드 완료 ({len(text_file):,}자)")
            with st.expander("파일 내용 미리보기"):
                preview = text_file[:2000]
                if len(text_file) > 2000:
                    preview += "\n\n... (이하 생략)"
                st.text(preview)
        except Exception as e:
            st.error(f"파일 읽기 실패: {e}")

st.divider()

source = st.radio(
    "사용할 입력", ["직접 입력", "업로드 파일"], horizontal=True,
)
text = (text_direct if source == "직접 입력" else text_file).strip()

if text:
    text_bytes = len(text.encode("utf-8"))
    if engine == "Gemini TTS (신규)":
        chunk_max = get_max_bytes(model)
        chunks_preview = split_text(text, max_bytes=chunk_max)
        limit_hint = f"청크 한도 {chunk_max:,}B"
    else:
        chunks_preview = split_text(text, max_sentence_bytes=max_sentence_bytes)
        limit_hint = "Cloud TTS 기본 청크"
    st.info(
        f"📝 총 {len(text):,}자 · {text_bytes:,} bytes · "
        f"{len(chunks_preview)}개 청크로 분할 예정 ({limit_hint})"
    )

    chunks_key = f"{engine}|{hash(tuple(chunks_preview))}"
    if st.session_state.get("chunks_key") != chunks_key:
        st.session_state.chunks_key = chunks_key
        st.session_state.chunk_audios = {}
        st.session_state.chunk_errors = {}
        st.session_state.gemini_key_idx = 0
        st.session_state.call_log = []

    if engine == "Gemini TTS (신규)":
        pool_sig = "|".join(gemini_key_pool)
    else:
        pool_sig = f"CLOUD:{active_key}"
    if st.session_state.get("pool_sig") != pool_sig:
        st.session_state.pool_sig = pool_sig
        st.session_state.gemini_key_idx = 0

    if "chunk_audios" not in st.session_state:
        st.session_state.chunk_audios = {}
    if "chunk_errors" not in st.session_state:
        st.session_state.chunk_errors = {}
    if "gemini_key_idx" not in st.session_state:
        st.session_state.gemini_key_idx = 0
    if "call_log" not in st.session_state:
        st.session_state.call_log = []

    if engine == "Gemini TTS (신규)" and st.session_state.gemini_key_idx >= len(gemini_key_pool):
        st.session_state.gemini_key_idx = 0

preview_btn = st.button(
    "🎧 미리듣기 (앞 200자)",
    use_container_width=True,
    disabled=not text,
)


def _audio_ext():
    return "wav" if engine == "Gemini TTS (신규)" else "mp3"


def _audio_mime():
    return "audio/wav" if engine == "Gemini TTS (신규)" else "audio/mp3"


if preview_btn and text:
    sample_text = text[:200]
    with st.spinner("미리듣기 생성 중..."):
        try:
            if engine == "Gemini TTS (신규)":
                audio = synthesize_gemini(
                    active_key, sample_text, model, voice_name, style_prompt,
                    seed=seed_value, temperature=temperature_value,
                )
            else:
                audio = synthesize(
                    active_key, sample_text, voice_name,
                    language_code=language_code,
                    speaking_rate=speaking_rate, pitch=pitch,
                )
            st.audio(audio, format=_audio_mime())
        except Exception as e:
            st.error(f"생성 실패: {e}")

if text and chunks_preview:
    st.divider()
    total_chunks = len(chunks_preview)
    done_count = sum(1 for i in range(total_chunks) if i in st.session_state.chunk_audios)

    st.subheader(f"🎙️ 청크별 생성 ({done_count}/{total_chunks} 완료)")

    ext = _audio_ext()
    mime = _audio_mime()

    if done_count == total_chunks and total_chunks > 0:
        all_parts = [st.session_state.chunk_audios[i] for i in range(total_chunks)]
        if engine == "Gemini TTS (신규)":
            merged = merge_wavs(all_parts)
        else:
            merged = b"".join(all_parts)
        st.success(f"🎉 모든 청크 완료 · 총 {len(merged):,} 바이트")
        st.audio(merged, format=mime)
        col_all, col_zip = st.columns(2)
        with col_all:
            st.download_button(
                f"💾 전체 {ext.upper()} (하나로 합치기)",
                merged,
                file_name=f"tts_output.{ext}",
                mime=mime,
                use_container_width=True,
                key="dl_merged",
            )
        with col_zip:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, part in enumerate(all_parts, 1):
                    zf.writestr(f"tts_part_{i:03d}.{ext}", part)
            buf.seek(0)
            st.download_button(
                "📦 ZIP (청크별 파일)",
                buf,
                file_name="tts_output.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_zip",
            )

    if done_count > 0:
        if st.button("🗑️ 전체 초기화 (다시 생성)", use_container_width=True):
            st.session_state.chunk_audios = {}
            st.session_state.chunk_errors = {}
            st.session_state.gemini_key_idx = 0
            st.rerun()

    st.caption(f"🔑 현재 사용 중인 Gemini 키: #{st.session_state.gemini_key_idx + 1}")

    log = st.session_state.call_log
    with st.expander(f"📡 모델 호출 로그 ({len(log)}회)", expanded=False):
        if not log:
            st.caption("아직 호출 기록이 없습니다.")
        else:
            col_clear, _ = st.columns([1, 3])
            with col_clear:
                if st.button("🗑️ 로그 지우기", key="clear_log"):
                    st.session_state.call_log = []
                    st.rerun()
            for entry in reversed(log):
                status_icon = "✅" if entry["status"] == "성공" else "❌"
                ttfb = entry.get("ttfb")
                ttfb_str = f"{ttfb:.1f}s" if ttfb is not None else "없음"
                line = (
                    f"{status_icon} `{entry['time']}` · 청크 {entry['chunk']} · "
                    f"키 #{entry['key']} · {entry['duration']:.1f}s · "
                    f"TTFB {ttfb_str} · {entry.get('events', 0)}이벤트 · "
                    f"{entry.get('pcm_bytes', 0):,}B · "
                    f"finish={entry.get('finish_reason') or '?'} · {entry['status']}"
                )
                if entry.get("error"):
                    line += f"\n  — {entry['error'][:200]}"
                st.markdown(line)

    last_call_by_chunk = {}
    for entry in log:
        last_call_by_chunk[entry["chunk"]] = entry

    for i, chunk in enumerate(chunks_preview):
        chunk_bytes = len(chunk.encode("utf-8"))
        preview_text = chunk[:40].replace("\n", " ")
        if len(chunk) > 40:
            preview_text += "…"

        is_done = i in st.session_state.chunk_audios
        has_error = i in st.session_state.chunk_errors
        icon = "✅" if is_done else ("❌" if has_error else "⏳")

        col_info, col_action = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"**{icon} 청크 {i + 1}/{total_chunks}** · "
                f"{chunk_bytes:,}B · `{preview_text}`"
            )
            last = last_call_by_chunk.get(i + 1)
            if last:
                ttfb = last.get("ttfb")
                ttfb_str = f"{ttfb:.1f}s" if ttfb is not None else "없음"
                st.caption(
                    f"📡 마지막 호출: {last['time']} · 키 #{last['key']} · "
                    f"{last['duration']:.1f}s · TTFB {ttfb_str} · "
                    f"{last.get('events', 0)}이벤트 · "
                    f"{last.get('pcm_bytes', 0):,}B · {last['status']}"
                )
            if has_error and not is_done:
                st.error(st.session_state.chunk_errors[i])
            with st.expander("📄 본문 보기"):
                st.text(chunk)
        with col_action:
            if is_done:
                st.download_button(
                    f"💾 다운로드",
                    st.session_state.chunk_audios[i],
                    file_name=f"tts_part_{i + 1:03d}.{ext}",
                    mime=mime,
                    key=f"dl_{i}",
                    use_container_width=True,
                )
                if st.button("🔄 재생성", key=f"regen_{i}", use_container_width=True):
                    del st.session_state.chunk_audios[i]
                    st.session_state.chunk_errors.pop(i, None)
                    st.rerun()
            else:
                btn_label = "🔁 재시도" if has_error else "▶ 생성"
                btn_type = "secondary" if has_error else "primary"
                if st.button(btn_label, key=f"gen_{i}", type=btn_type, use_container_width=True):
                    start_key = st.session_state.gemini_key_idx
                    started_at = time.monotonic()
                    ts = datetime.now().strftime("%H:%M:%S")
                    call_stats = {}
                    with st.spinner(f"청크 {i + 1} 생성 중..."):
                        try:
                            if engine == "Gemini TTS (신규)":
                                wavs, new_idx = synthesize_gemini_chunk(
                                    gemini_key_pool,
                                    st.session_state.gemini_key_idx,
                                    chunk, model, voice_name,
                                    style_prompt=style_prompt,
                                    seed=seed_value,
                                    temperature=temperature_value,
                                    stats=call_stats,
                                )
                                st.session_state.gemini_key_idx = new_idx
                                audio_bytes = merge_wavs(wavs)
                                used_key = new_idx + 1
                            else:
                                audio_bytes = synthesize(
                                    active_key, chunk, voice_name,
                                    language_code=language_code,
                                    speaking_rate=speaking_rate, pitch=pitch,
                                )
                                used_key = 1
                            st.session_state.chunk_audios[i] = audio_bytes
                            st.session_state.chunk_errors.pop(i, None)
                            st.session_state.call_log.append({
                                "time": ts,
                                "chunk": i + 1,
                                "key": used_key,
                                "duration": time.monotonic() - started_at,
                                "status": "성공",
                                "error": None,
                                "ttfb": call_stats.get("ttfb"),
                                "events": call_stats.get("events", 0),
                                "pcm_bytes": call_stats.get("pcm_bytes", 0),
                                "finish_reason": call_stats.get("finish_reason"),
                            })
                            st.rerun()
                        except QuotaExceeded as e:
                            st.session_state.chunk_errors[i] = (
                                f"🚫 모든 API 키의 할당량이 소진되었습니다. "
                                f"사이드바에 백업 키를 추가하거나 내일 다시 시도하세요.\n\n{e}"
                            )
                            st.session_state.call_log.append({
                                "time": ts,
                                "chunk": i + 1,
                                "key": start_key + 1,
                                "duration": time.monotonic() - started_at,
                                "status": "할당량 초과",
                                "error": str(e),
                                "ttfb": call_stats.get("ttfb"),
                                "events": call_stats.get("events", 0),
                                "pcm_bytes": call_stats.get("pcm_bytes", 0),
                                "finish_reason": call_stats.get("finish_reason"),
                            })
                            st.rerun()
                        except Exception as e:
                            st.session_state.chunk_errors[i] = str(e)
                            st.session_state.call_log.append({
                                "time": ts,
                                "chunk": i + 1,
                                "key": start_key + 1,
                                "duration": time.monotonic() - started_at,
                                "status": "실패",
                                "error": str(e),
                                "ttfb": call_stats.get("ttfb"),
                                "events": call_stats.get("events", 0),
                                "pcm_bytes": call_stats.get("pcm_bytes", 0),
                                "finish_reason": call_stats.get("finish_reason"),
                            })
                            st.rerun()
