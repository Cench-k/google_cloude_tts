def _read_txt(file):
    content = file.read()
    if isinstance(content, bytes):
        for enc in ("utf-8", "cp949", "euc-kr"):
            try:
                return content.decode(enc)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")
    return content


def _read_docx(file):
    from docx import Document

    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


def _read_pdf(file):
    from pypdf import PdfReader

    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return _read_txt(uploaded_file)
    if name.endswith(".docx"):
        return _read_docx(uploaded_file)
    if name.endswith(".pdf"):
        return _read_pdf(uploaded_file)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {name}")
