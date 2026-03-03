import os
import re
import urllib.request
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
import google.generativeai as genai
from time import sleep

# --- 설정 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-1.5-pro" # 최고 성능 모델
DOCS_DIR = "docs"
IMAGE_DIR = os.path.join(DOCS_DIR, "images")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# 시스템 프롬프트 (최종 확정본)
SYSTEM_PROMPT = """
You are a world-class AI/ML paper analysis expert. I am providing you with the main text of an academic paper.

**[Analysis Guidelines]**
- Translate the content into Korean, but keep commonly used technical terms in their original English.
- If a section's required information is not explicitly mentioned, indicate it as "논문에서 명시되지 않음".
- No speculation or hallucinations.

---
**[Output Format]**

#### 2. 🎯 연구 목적 (Why)
* 이 연구가 해결하고자 하는 기존의 문제점이나 배경은 무엇인가?

#### 3. 🛠️ 연구 방법론 (How)
* **Core Idea:** (핵심 아이디어 상세 설명)
* **Model Architecture:** (모델 구조 및 주요 컴포넌트)

#### 4. 💡 주요 결과 (What)
* **Dataset & Evaluation:** (데이터셋 및 평가 방법)
* **Experimental Results:** (주요 수치 및 성능 개선 폭 등)
* **Key Findings:** (가장 중요한 발견 3가지)

#### 5. ⚠️ 한계점 및 향후 과제
* **Limitations:** * **(1) 논문에서 저자가 밝힌 한계점:**
  * **(2) 분석가(AI)의 견해:** * **Future Work:** * **(1) 논문에서 저자가 밝힌 향후 과제:**
  * **(2) 분석가(AI)의 견해:** #### 6. 🧠 핵심 인사이트
* **Significance:** (이 연구가 분야에 미치는 의미와 영향)
* **Practical Application:** (실무나 연구에 즉시 적용하거나 참고해 볼 만한 아이디어 1가지)

#### 7. 🔗 기타
* **Code:** (코드 링크 기재, 없으면 생략)
* **Demo:** (데모 링크 기재, 없으면 생략)

#### 8. 📑 논문 전체 목차
* **Table of Contents:** (논문 본문을 분석하여 메인 섹션 헤더 추출)
"""

def parse_arxiv_id(url: str):
    match = re.search(r'arxiv\.org/(?:abs|pdf|html)/([a-zA-Z0-9.\-]+)', url)
    return match.group(1) if match else None

def fetch_arxiv_metadata(arxiv_id: str):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        response = urllib.request.urlopen(url)
        root = ET.fromstring(response.read())
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        if entry is None: return None
        
        return {
            "title": entry.find('atom:title', ns).text.replace('\n', ' ').strip(),
            "abstract": entry.find('atom:summary', ns).text.replace('\n', ' ').strip(),
            "authors": ", ".join([a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]),
            "year": entry.find('atom:published', ns).text[:4]
        }
    except:
        return None

def download_pdf_safely(url: str, output_path: str):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as f:
            f.write(response.read())
        return True
    except:
        return False

def extract_and_crop_pdf(pdf_path: str, paper_id: str):
    doc = fitz.open(pdf_path)
    full_text = ""
    image_paths = []
    
    for page_num, page in enumerate(doc):
        full_text += page.get_text("text") + "\n"
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            base_image = doc.extract_image(img[0])
            if len(base_image["image"]) < 10000: continue # 너무 작은 노이즈 아이콘 제외
            
            img_filename = f"{paper_id}_p{page_num+1}_img{img_index}.{base_image['ext']}"
            with open(os.path.join(IMAGE_DIR, img_filename), "wb") as f:
                f.write(base_image["image"])
            image_paths.append(img_filename)
    doc.close()

    # 토큰 다이어트: References 이후 절단
    match = re.search(r'\n\s*(References|Bibliography|REFERENCES)\s*\n', full_text)
    cropped_text = full_text[:match.start()] if match else full_text
    return cropped_text, image_paths

def process_paper(source_url: str):
    print(f"[{source_url}] 분석 시작...")
    arxiv_id = parse_arxiv_id(source_url)
    paper_id = arxiv_id.replace('.', '_') if arxiv_id else f"paper_{hash(source_url) % 10000}"
    md_filename = os.path.join(DOCS_DIR, f"{paper_id}.md")

    # 이미 분석된 논문이면 스킵
    if os.path.exists(md_filename):
        print("이미 분석된 논문입니다. 스킵합니다.")
        return

    metadata = None
    pdf_path = f"temp_{paper_id}.pdf"

    if arxiv_id:
        metadata = fetch_arxiv_metadata(arxiv_id)
        download_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    else:
        download_url = source_url

    if not download_pdf_safely(download_url, pdf_path):
        print("PDF 다운로드 실패")
        return

    cropped_text, images = extract_and_crop_pdf(pdf_path, paper_id)
    os.remove(pdf_path) # 임시 파일 삭제

    # Gemini API 호출
    model = genai.GenerativeModel(MODEL_NAME)
    api_input = SYSTEM_PROMPT + "\n\n--- PAPER CONTENT ---\n" + cropped_text
    try:
        response = model.generate_content(api_input)
        analysis_result = response.text
    except Exception as e:
        print(f"API 에러: {e}")
        return

    # 마크다운 작성
    with open(md_filename, "w", encoding="utf-8") as f:
        if metadata:
            f.write(f"# 📄 {metadata['title']}\n\n")
            f.write(f"* **저자 / 기관 / 발행년도:** {metadata['authors']} / {metadata['year']}\n")
            f.write(f"* **원문 링크:** [{source_url}]({source_url})\n\n")
            f.write(f"#### 1. 📖 Abstract\n* **Original:** {metadata['abstract']}\n\n---\n")
        else:
            f.write(f"# 📄 [논문 분석: {paper_id}]\n\n* **원문 링크:** [{source_url}]({source_url})\n\n---\n")
        
        f.write(analysis_result)
        f.write("\n\n---\n#### 🖼️ 추출된 주요 그림 및 표\n\n")
        for img in images:
            f.write(f"![{img}](images/{img})\n\n")

    print(f"완료: {md_filename}")

if __name__ == "__main__":
    if os.path.exists("paper_links.txt"):
        with open("paper_links.txt", "r") as f:
            for line in f:
                url = line.strip()
                if url:
                    process_paper(url)
                    sleep(3) # API Rate Limit 방지
