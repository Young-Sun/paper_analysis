import os
import re
import urllib.request
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
from google import genai
from time import sleep

# --- 설정 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

DOCS_DIR = "docs"
IMAGE_DIR = os.path.join(DOCS_DIR, "images")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# 시스템 프롬프트 (최신 버전 유지)
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

def get_latest_model(mode="free"):
    try:
        available_models = [m.name for m in client.models.list()]
        target_models = [m for m in available_models if 'gemini' in m and ('pro' if mode == 'pro' else 'flash') in m and 'vision' not in m]
        target_models.sort(reverse=True)
        if target_models:
            return target_models[0].replace('models/', '')
    except:
        pass
    return "gemini-2.0-flash" if mode == "free" else "gemini-2.0-pro"

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

def extract_figures_and_captions(pdf_path: str, paper_id: str):
    doc = fitz.open(pdf_path)
    figure_data = []
    
    for page_num, page in enumerate(doc):
        text_blocks = page.get_text("blocks")
        page_images = page.get_images(full=True)
        
        for block in text_blocks:
            block_text = block[4].strip()
            # Fig 1, Figure 2 등의 패턴 매칭
            match = re.search(r'^\s*(Fig|Figure|Figs|FIG|FIGURE)\.?\s*(\d+)', block_text, re.IGNORECASE)
            
            if match:
                fig_number = match.group(2)
                full_caption = block_text.replace('\n', ' ')
                b_rect = fitz.Rect(block[:4])
                
                # 검색 영역 확장: 캡션 위로 500pt(거의 반 페이지), 아래로 50pt 확장
                search_rect = fitz.Rect(b_rect.x0 - 50, b_rect.y0 - 500, b_rect.x1 + 50, b_rect.y1 + 50)
                
                target_xref = None
                max_area = 0
                
                for img in page_images:
                    xref = img[0]
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        img_rect = img_rects[0]
                        # 검색 영역 내에 있고, 일정 크기 이상 중 가장 큰 이미지 선택
                        if search_rect.intersects(img_rect):
                            area = img_rect.width * img_rect.height
                            if area > max_area and area > 5000:
                                max_area = area
                                target_xref = xref
                
                if target_xref:
                    img_filename = f"{paper_id}_Fig{fig_number}_p{page_num+1}.png"
                    img_filepath = os.path.join(IMAGE_DIR, img_filename)
                    
                    try:
                        # 픽스맵 추출
                        pix = fitz.Pixmap(doc, target_xref)
                        
                        # [배경색 보정 핵심] 흰색 배경 도화지 생성
                        img_rgb = fitz.Pixmap(fitz.csRGB, pix.width, pix.height, 0)
                        img_rgb.clear_with_white()
                        
                        # 원본이 Gray나 CMYK일 경우를 대비해 RGB로 변환하여 복사
                        if pix.colorspace.n < 3:
                            temp_pix = fitz.Pixmap(fitz.csRGB, pix)
                            img_rgb.copy(temp_pix, (0, 0, pix.width, pix.height))
                        else:
                            img_rgb.copy(pix, (0, 0, pix.width, pix.height))
                        
                        img_rgb.save(img_filepath)
                        figure_data.append((img_filename, full_caption))
                        
                        pix = None
                        img_rgb = None
                    except:
                        pass
    doc.close()
    return figure_data

def process_paper(source_url: str, mode: str = "free"):
    arxiv_id = parse_arxiv_id(source_url)
    paper_id = arxiv_id.replace('.', '_') if arxiv_id else f"paper_{hash(source_url) % 10000}"
    md_filename = os.path.join(DOCS_DIR, f"{paper_id}.md")

    # 분석 건너뛰기 로직 (수정 시 잠시 주석 처리 가능)
    if os.path.exists(md_filename):
        print(f"이미 분석된 논문입니다. 스킵합니다: {source_url}")
        return

    model_name = get_latest_model(mode)
    print(f"\n[{source_url}] 분석 시작... (모델: {model_name} / 모드: {mode.upper()})")

    pdf_path = f"temp_{paper_id}.pdf"
    if not download_pdf_safely(f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else source_url, pdf_path):
        return

    # 1. Figure 및 캡션 추출
    figure_data = extract_figures_and_captions(pdf_path, paper_id)
    
    # 2. 본문 텍스트 추출 (References 절단)
    doc = fitz.open(pdf_path)
    full_text = "".join([p.get_text() for p in doc])
    match = re.search(r'\n\s*(References|Bibliography|REFERENCES)\s*\n', full_text)
    cropped_text = full_text[:match.start()] if match else full_text
    doc.close()
    os.remove(pdf_path)

    # 3. Gemini API 호출
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=SYSTEM_PROMPT + "\n\n--- PAPER CONTENT ---\n" + cropped_text,
        )
        analysis_result = response.text
    except Exception as e:
        print(f"API 에러: {e}")
        return

    # 4. 결과 저장
    metadata = fetch_arxiv_metadata(arxiv_id) if arxiv_id else None
    with open(md_filename, "w", encoding="utf-8") as f:
        if metadata:
            f.write(f"# 📄 {metadata['title']}\n\n* **저자:** {metadata['authors']} ({metadata['year']})\n* **원문:** [{source_url}]({source_url})\n\n---\n")
        else:
            f.write(f"# 📄 [논문 분석: {paper_id}]\n\n* **원문:** [{source_url}]({source_url})\n\n---\n")
        
        f.write(analysis_result)
        f.write("\n\n---\n#### 🖼️ 추출된 주요 그림(Figures)\n\n")
        
        for img_name, caption in figure_data:
            f.write(f"![{img_name}](images/{img_name})\n\n")
            f.write(f"> **{caption}**\n\n<br>\n\n")

    print(f"✅ 분석 완료: {md_filename}")

if __name__ == "__main__":
    if os.path.exists("paper_links.txt"):
        with open("paper_links.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = [p.strip() for p in line.split(',')]
                process_paper(parts[0], parts[1].lower() if len(parts) > 1 else 'free')
                sleep(3)
