import io
import os
import zipfile

import streamlit as st
from dotenv import load_dotenv

from file_utils import read_file
from tts_utils import list_voices, split_text, synthesize, synthesize_long

load_dotenv()

st.set_page_config(page_title="Google TTS 도우미", page_icon="🔊", layout="wide")
st.title("🔊 Google TTS 도우미")


def _get_api_key():
    try:
        if "GOOGLE_TTS_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_TTS_API_KEY"]
    except Exception:
        pass
    return os.getenv("GOOGLE_TTS_API_KEY")


api_key = _get_api_key()
if not api_key:
    st.error(
        "GOOGLE_TTS_API_KEY가 설정되지 않았습니다. "
        "로컬: `.env` 파일, 배포: Streamlit Secrets에 등록해주세요."
    )
    st.stop()


@st.cache_data(ttl=3600, show_spinner="목소리 목록 로딩 중...")
def get_voices(key, lang):
    return list_voices(key, lang)


with st.sidebar:
    st.header("⚙️ 설정")
    language_code = st.selectbox(
        "언어",
        ["ko-KR", "en-US", "ja-JP", "zh-CN"],
        index=0,
    )

    try:
        voices = get_voices(api_key, language_code)
    except Exception as e:
        st.error(f"목소리 목록 로드 실패: {e}")
        st.stop()

    if not voices:
        st.error("사용 가능한 목소리가 없습니다.")
        st.stop()

    tiers = sorted(
        {
            v["name"].split("-")[2]
            for v in voices
            if len(v["name"].split("-")) >= 3
        }
    )
    default_tiers = [t for t in ["Neural2", "Wavenet"] if t in tiers] or tiers
    tier_filter = st.multiselect("목소리 등급", tiers, default=default_tiers)

    filtered = (
        [v for v in voices if any(t in v["name"] for t in tier_filter)]
        if tier_filter
        else voices
    )
    if not filtered:
        st.warning("필터에 맞는 목소리가 없습니다.")
        filtered = voices

    voice_options = {
        f"{v['name']} · {v['ssmlGender']}": v["name"] for v in filtered
    }
    voice_label = st.selectbox("목소리 선택", list(voice_options.keys()))
    voice_name = voice_options[voice_label]

    speaking_rate = st.slider("속도", 0.25, 4.0, 1.0, 0.05)
    pitch = st.slider("피치", -20.0, 20.0, 0.0, 1.0)

    st.caption("💡 Neural2 > Wavenet > Standard 순으로 자연스럽습니다.")

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
    "사용할 입력",
    ["직접 입력", "업로드 파일"],
    horizontal=True,
)
text = (text_direct if source == "직접 입력" else text_file).strip()

if text:
    chunks_preview = split_text(text)
    st.info(
        f"📝 총 {len(text):,}자 · {len(chunks_preview)}개 청크로 분할 예정"
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

if preview_btn and text:
    with st.spinner("미리듣기 생성 중..."):
        try:
            audio = synthesize(
                api_key,
                text[:200],
                voice_name,
                language_code=language_code,
                speaking_rate=speaking_rate,
                pitch=pitch,
            )
            st.audio(audio, format="audio/mp3")
        except Exception as e:
            st.error(f"생성 실패: {e}")

if generate_btn and text:
    progress = st.progress(0.0, "준비 중...")
    status = st.empty()

    def update(done, total):
        progress.progress(done / total, f"{done}/{total} 청크 처리 중...")

    try:
        merged, parts = synthesize_long(
            api_key,
            text,
            voice_name,
            language_code=language_code,
            speaking_rate=speaking_rate,
            pitch=pitch,
            progress_cb=update,
        )
        progress.progress(1.0, "완료!")
        status.success(f"✅ 생성 완료 · {len(parts)}개 청크 · 총 {len(merged):,} 바이트")

        st.audio(merged, format="audio/mp3")

        if save_mode == "하나의 파일로 합치기":
            st.download_button(
                "💾 MP3 다운로드",
                merged,
                file_name="tts_output.mp3",
                mime="audio/mp3",
                use_container_width=True,
            )
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, part in enumerate(parts, 1):
                    zf.writestr(f"tts_part_{i:03d}.mp3", part)
            buf.seek(0)
            st.download_button(
                "💾 ZIP 다운로드",
                buf,
                file_name="tts_output.zip",
                mime="application/zip",
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"생성 실패: {e}")
