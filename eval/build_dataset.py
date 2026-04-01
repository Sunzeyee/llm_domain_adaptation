import re
import json
import fitz  # PyMuPDF

PDF_PATH = "../data/raw/mianzhanixi.pdf"
OUTPUT_PATH = "../data/processed/test.json"


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


def clean_text(text):
    # 去掉奇怪换行（关键🔥）
    text = text.replace("\r", "")
    text = re.sub(r"\n+", "\n", text)

    # 修复被截断的英文换行
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text


def extract_qa(text):
    """
    只保留：
    1. 带编号
    2. 是“问句”
    """

    pattern = r"\d+\.(.*?)\n(.*?)(?=\n\d+\.|$)"
    matches = re.findall(pattern, text, re.S)

    qa_list = []

    for q, a in matches:
        q = q.strip()
        a = a.strip()

        # ❗核心过滤（非常重要）
        if not q.endswith("？"):
            continue

        if len(q) < 5 or len(a) < 20:
            continue

        # 去掉 PS 垃圾
        a = re.sub(r"PS：.*?\n", "", a)

        qa_list.append({
            "question": q,
            "answer": a
        })

    return qa_list


def build_dataset():
    print("📄 Loading PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("🧹 Cleaning text...")
    text = clean_text(text)

    print("🔍 Extracting QA...")
    qa_list = extract_qa(text)

    print(f"✅ Valid QA pairs: {len(qa_list)}")

    print("💾 Saving dataset...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    print("✅ Done!")


if __name__ == "__main__":
    build_dataset()