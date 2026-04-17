import io
import os
import zipfile

import streamlit as st
from dotenv import load_dotenv

from file_utils import read_file
from gemini_tts import (
    GEMINI_MODELS,
    GEMINI_VOICES,
    QuotaExceeded,
    get_max_bytes,
    synthesize_gemini,
    synthesize_gemini_long,
)
from tts_utils import list_voices, split_text, synthesize, synthesize_long

load_dotenv()

st.set_page_config(page_title="Google TTS 도우미", page_icon="🔊", layout="wide")
st.title("🔊 Google TTS 도우미")


def _get_key(name, fallback=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    val = os.getenv(name)
    if val:
        return val
    return fallback


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

cloud_key = _get_key("GOOGLE_TTS_API_KEY")
gemini_key = _get_key("GEMINI_API_KEY", fallback=cloud_key)


def _parse_backup_keys(raw: str) -> list:
    if not raw:
        return []
    parts = []
    for line in raw.replace(",", "\n").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            parts.append(line)
    return parts


backup_keys_env = _get_key("GEMINI_API_KEYS_BACKUP", fallback="") or ""

if engine == "Gemini TTS (신규)":
    if not gemini_key:
        st.error(
            "Gemini API 키가 설정되지 않았습니다. "
            "`.env`에 `GEMINI_API_KEY` 또는 `GOOGLE_TTS_API_KEY`를 등록해주세요."
        )
        st.stop()
    active_key = gemini_key
else:
    if not cloud_key:
        st.error(
            "`GOOGLE_TTS_API_KEY`가 설정되지 않았습니다."
        )
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

        with st.expander("🔑 백업 API 키 (할당량 초과 시 자동 전환)"):
            backup_raw = st.text_area(
                "추가 키 (한 줄에 하나)",
                value=backup_keys_env,
                height=100,
                placeholder="AIzaSy...\nAIzaSy...",
                help="주 키 할당량이 소진되면 위에서부터 순서대로 다음 키로 전환합니다.",
            )
        backup_keys = _parse_backup_keys(backup_raw)
        gemini_key_pool = [active_key] + backup_keys
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
    if engine == "Gemini TTS (신규)":
        chunks_preview = split_text(text, max_bytes=4000)
    else:
        chunks_preview = split_text(text, max_sentence_bytes=max_sentence_bytes)
    st.info(f"📝 총 {len(text):,}자 · {len(chunks_preview)}개 청크로 분할 예정")

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

if generate_btn and text:
    progress = st.progress(0.0, "준비 중...")
    status = st.empty()
    rotate_log = st.empty()

    rotation_events = []

    def update_gemini(done, total, active_idx):
        msg = f"{done}/{total} 청크 처리 중... (키 #{active_idx})"
        progress.progress(done / total, msg)

    def update_cloud(done, total):
        progress.progress(done / total, f"{done}/{total} 청크 처리 중...")

    def on_rotate(old_idx, new_idx, err_msg):
        rotation_events.append(f"⚠️ 키 #{old_idx + 1} 할당량 초과 → 키 #{new_idx + 1}로 전환")
        rotate_log.warning("\n".join(rotation_events))

    try:
        if engine == "Gemini TTS (신규)":
            merged, parts = synthesize_gemini_long(
                gemini_key_pool, text, model, voice_name, style_prompt,
                seed=seed_value, temperature=temperature_value,
                progress_cb=update_gemini,
                rotate_cb=on_rotate,
            )
        else:
            merged, parts = synthesize_long(
                active_key, text, voice_name,
                language_code=language_code,
                speaking_rate=speaking_rate, pitch=pitch,
                max_sentence_bytes=max_sentence_bytes,
                progress_cb=update_cloud,
            )
        progress.progress(1.0, "완료!")
        status.success(f"✅ 생성 완료 · {len(parts)}개 청크 · 총 {len(merged):,} 바이트")

        st.audio(merged, format=_audio_mime())

        ext = _audio_ext()
        if save_mode == "하나의 파일로 합치기":
            st.download_button(
                f"💾 {ext.upper()} 다운로드",
                merged,
                file_name=f"tts_output.{ext}",
                mime=_audio_mime(),
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
    except QuotaExceeded as e:
        progress.empty()
        st.error(
            "🚫 **모든 API 키의 할당량이 소진되었습니다.**\n\n"
            f"{e}\n\n"
            "**해결 방법**\n"
            "- 사이드바 '🔑 백업 API 키'에 추가 키를 등록하고 다시 시도\n"
            "- AI Studio에서 새 키 발급 (https://aistudio.google.com/apikey)\n"
            "- 무료 한도는 자정(태평양 시각)에 초기화됩니다",
            icon="🚫",
        )
    except Exception as e:
        st.error(f"생성 실패: {e}")
