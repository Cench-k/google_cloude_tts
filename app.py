import io
import zipfile

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
from tts_utils import list_voices, split_text, synthesize, synthesize_long

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
                try:
                    sample_audio = get_gemini_sample(active_key, model, voice_name)
                    st.audio(sample_audio, format="audio/wav")
                except Exception as e:
                    st.error(f"샘플 생성 실패: {e}")

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
    with st.expander("🔍 청크 경계 확인 (각 청크 끝 40자 미리보기)"):
        for i, chunk in enumerate(chunks_preview, 1):
            chunk_bytes = len(chunk.encode("utf-8"))
            tail = chunk[-40:].replace("\n", "↵")
            ends_on_break = chunk.rstrip().endswith((".", "!", "?", "。", "！", "？"))
            mark = "✅" if ends_on_break else "⚠️"
            st.write(
                f"{mark} **청크 {i}** ({chunk_bytes:,}B) "
                f"··· `{tail}`"
            )

col1, col2 = st.columns(2)
with col1:
    preview_btn = st.button(
        "🎧 미리듣기 (앞 200자)",
        use_container_width=True,
        disabled=not text,
    )
with col2:
    generate_btn = st.button(
        "🎵 전체 생성",
        type="primary",
        use_container_width=True,
        disabled=not text,
    )

save_mode = st.radio(
    "저장 방식",
    ["하나의 파일로 합치기", "청크별 개별 파일 (ZIP)"],
    horizontal=True,
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

if "tts_job" not in st.session_state:
    st.session_state.tts_job = None

if generate_btn and text:
    if engine == "Gemini TTS (신규)":
        job_chunks = split_text(text, max_bytes=get_max_bytes(model))
    else:
        job_chunks = split_text(text, max_sentence_bytes=max_sentence_bytes)

    if not job_chunks:
        st.error("입력 텍스트가 비어 있습니다.")
    else:
        st.session_state.tts_job = {
            "engine": engine,
            "chunks": job_chunks,
            "done_parts": [],
            "key_idx": 0,
            "rotation_events": [],
            "params": {
                "model": model,
                "voice_name": voice_name,
                "style_prompt": style_prompt,
                "seed": seed_value,
                "temperature": temperature_value,
                "keys": gemini_key_pool if engine == "Gemini TTS (신규)" else [active_key],
                "language_code": language_code,
                "speaking_rate": speaking_rate,
                "pitch": pitch,
            },
            "status": "running",
            "error": None,
        }
        st.rerun()

job = st.session_state.tts_job

if job and job["status"] in ("running", "error"):
    total = len(job["chunks"])
    done = len(job["done_parts"])
    progress = st.progress(done / total if total else 0,
                           f"{done}/{total} 청크 완료 · 다음 청크 처리 중...")
    if job["rotation_events"]:
        st.warning("\n".join(job["rotation_events"]))

    col_cancel, col_retry = st.columns(2)
    with col_cancel:
        if st.button("❌ 생성 취소", use_container_width=True):
            st.session_state.tts_job = None
            st.rerun()
    with col_retry:
        if job["status"] == "error" and st.button("🔁 이어서 재시도", use_container_width=True, type="primary"):
            job["status"] = "running"
            job["error"] = None
            st.session_state.tts_job = job
            st.rerun()

    if job["status"] == "error":
        st.error(f"청크 {done + 1} 실패: {job['error']}")
    else:
        chunk = job["chunks"][done]
        params = job["params"]

        def _on_rotate(old_i, new_i, err_msg):
            job["rotation_events"].append(
                f"⚠️ 키 #{old_i + 1} 할당량 초과 → 키 #{new_i + 1}로 전환"
            )

        try:
            if job["engine"] == "Gemini TTS (신규)":
                wavs, new_idx = synthesize_gemini_chunk(
                    params["keys"], job["key_idx"], chunk,
                    params["model"], params["voice_name"],
                    style_prompt=params["style_prompt"],
                    seed=params["seed"], temperature=params["temperature"],
                    rotate_cb=_on_rotate,
                )
                job["key_idx"] = new_idx
                merged_chunk = merge_wavs(wavs)
                job["done_parts"].append(merged_chunk)
            else:
                audio = synthesize(
                    params["keys"][0], chunk, params["voice_name"],
                    language_code=params["language_code"],
                    speaking_rate=params["speaking_rate"], pitch=params["pitch"],
                )
                job["done_parts"].append(audio)

            if len(job["done_parts"]) >= total:
                job["status"] = "done"
            st.session_state.tts_job = job
            st.rerun()
        except QuotaExceeded as e:
            job["status"] = "error"
            job["error"] = (
                "🚫 모든 API 키의 할당량이 소진되었습니다. "
                "사이드바에 백업 키를 추가하거나 내일 다시 시도하세요.\n\n"
                f"{e}"
            )
            st.session_state.tts_job = job
            st.rerun()
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            st.session_state.tts_job = job
            st.rerun()

if job and job["status"] == "done":
    total = len(job["chunks"])
    parts = job["done_parts"]
    engine_finished = job["engine"]
    ext = "wav" if engine_finished == "Gemini TTS (신규)" else "mp3"
    mime = "audio/wav" if engine_finished == "Gemini TTS (신규)" else "audio/mp3"

    if engine_finished == "Gemini TTS (신규)":
        merged = merge_wavs(parts)
    else:
        merged = b"".join(parts)

    st.success(f"✅ 생성 완료 · {total}개 청크 · 총 {len(merged):,} 바이트")
    st.audio(merged, format=mime)

    if save_mode == "하나의 파일로 합치기":
        st.download_button(
            f"💾 {ext.upper()} 다운로드",
            merged,
            file_name=f"tts_output.{ext}",
            mime=mime,
            use_container_width=True,
        )
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, part in enumerate(parts, 1):
                zf.writestr(f"tts_part_{i:03d}.{ext}", part)
        buf.seek(0)
        st.download_button(
            "💾 ZIP 다운로드",
            buf,
            file_name="tts_output.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if st.button("🗑️ 결과 지우고 새로 시작", use_container_width=True):
        st.session_state.tts_job = None
        st.rerun()
