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

# 시스템 프롬프트
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
        if mode == "pro":
            target_models = [m for m in available_models if 'gemini' in m and 'pro' in m and 'vision' not in m]
        else:
            target_models = [m for m in available_models if 'gemini' in m and 'flash' in m and 'vision' not in m]
        target_models.sort(reverse=True)
        if target_models:
            return target_models[0].replace('models/', '')
    except:
        pass
    return "gemini-2.5-pro" if mode == "pro" else "gemini-2.5-flash"

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

# --- 핵심 수정 부분: Figures 및 캡션 지능적 추출 ---
def extract_figures_and_captions(pdf_path: str, paper_id: str):
    """
    페이지 내 'Fig.' 텍스트 위치를 찾고, 그 주변의 그림 영역을 캡처하여 저장합니다.
    동시에 해당 Figure의 캡션 전체 텍스트를 추출합니다.
    """
    doc = fitz.open(pdf_path)
    figure_data = [] # (이미지파일명, 캡션텍스트) 튜플 리스트
    
    for page_num, page in enumerate(doc):
        # 1. 캡션 텍스트 블록 찾기 ("Fig. 1" 등)
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            block_text = block[4].strip()
            # 정규식으로 Figure 캡션 시작 부분 탐색 (예: Fig. 1, Figure 1, Figs. 1, FIGS. 1)
            match = re.search(r'^\s*(Fig|Figure|Figs|FIG|FIGURE)\.?\s*(\d+)', block_text, re.IGNORECASE)
            
            if match:
                fig_number = match.group(2)
                full_caption = block_text.replace('\n', ' ') # 캡션 내 줄바꿈 제거
                
                # 2. 캡션 블록 위치 (b_rect) 기반으로 주변 그림 영역 탐색
                b_rect = fitz.Rect(block[:4])
                
                # 캡션 바로 위(또는 아래) 영역 설정 (그림이 있을 것으로 예상되는 곳)
                # 캡션 블록의 높이만큼 위로 영역 확장
                search_rect = fitz.Rect(b_rect.x0, b_rect.y0 - b_rect.height*2, b_rect.x1, b_rect.y0)
                
                # 3. 해당 영역 내의 실제 그림(Image/Drawing) 객체 탐색
                page_images = page.get_images(full=True)
                target_image = None
                
                for img in page_images:
                    xref = img[0]
                    # 이미지 객체의 페이지 내 좌표 영역(rect) 가져오기
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        img_rect = img_rects[0]
                        # 캡션 주변 검색 영역(search_rect)과 실제 이미지 영역(img_rect)이 겹치는지 확인
                        if search_rect.intersects(img_rect):
                            target_image = xref
                            break # 첫 번째 매칭되는 그림 선택
                
                # 4. 그림 매칭 성공 시, 배경색 수정 후 저장
                if target_image:
                    img_filename = f"{paper_id}_Fig{fig_number}_p{page_num+1}.png"
                    img_filepath = os.path.join(IMAGE_DIR, img_filename)
                    
                    try:
                        # 배경색 수정 로직 (투명 -> 흰색)
                        # 원본 이미지 정보를 픽스맵(Pixmap)으로 가져옴
                        pix = fitz.Pixmap(doc, target_image)
                        
                        # 투명도(Alpha) 채널이 있는 경우 (배경이 검게 나오는 원인)
                        if pix.alpha:
                            # 흰색 배경의 새로운 픽스맵 생성
                            wh_pix = fitz.Pixmap(fitz.csRGB, pix.width, pix.height, 0)
                            wh_pix.clear_with_white() # 전체를 흰색으로 채움
                            
                            # 원본 이미지를 흰색 배경 위에 덮어쓰기 (투명한 부분이 흰색으로 보임)
                            wh_pix.copy(pix, (0, 0, pix.width, pix.height))
                            pix = wh_pix # 교체
                        
                        # 최종 이미지 저장
                        pix.save(img_filepath)
                        pix = None # 메모리 해제
                        
                        figure_data.append((img_filename, full_caption))
                        # print(f"    -> Fig {fig_number} 추출 성공: {img_filename}")
                    except Exception as e:
                        print(f"    -> Fig {fig_number} 저장 실패: {e}")
    
    doc.close()
    return figure_data

# --- 토큰 다이어트 함수 ---
def crop_references(pdf_path: str):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    doc.close()
    
    match = re.search(r'\n\s*(References|Bibliography|REFERENCES)\s*\n', full_text)
    cropped_text = full_text[:match.start()] if match else full_text
    return cropped_text

def process_paper(source_url: str, mode: str = "free"):
    arxiv_id = parse_arxiv_id(source_url)
    paper_id = arxiv_id.replace('.', '_') if arxiv_id else f"paper_{hash(source_url) % 10000}"
    md_filename = os.path.join(DOCS_DIR, f"{paper_id}.md")

    if os.path.exists(md_filename):
        print(f"이미 분석된 논문입니다. 스킵합니다: {source_url}")
        return

    model_name = get_latest_model(mode)
    print(f"\n[{source_url}] 분석 시작... (선택된 모델: {model_name} / 모드: {mode.upper()})")

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

    # 1. Figures 및 캡션 지능적 추출 (배경색 수정 포함)
    print(" -> 정식 Figure 및 캡션 매칭 추출 진행 중...")
    figure_data = extract_figures_and_captions(pdf_path, paper_id)
    
    # 2. 토큰 다이어트 (본문 텍스트 추출)
    cropped_text = crop_references(pdf_path)
    
    os.remove(pdf_path) # 임시 파일 삭제

    # 3. Gemini API 호출
    print(f" -> Gemini API 호출 중...")
    api_input = SYSTEM_PROMPT + "\n\n--- PAPER CONTENT ---\n" + cropped_text
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=api_input,
        )
        analysis_result = response.text
    except Exception as e:
        print(f"API 에러: {e}")
        return

    # 4. 마크다운 작성 및 조립
    print(" -> 마크다운 보고서 최종 조립 중...")
    with open(md_filename, "w", encoding="utf-8") as f:
        if metadata:
            f.write(f"# 📄 {metadata['title']}\n\n")
            f.write(f"* **저자 / 기관 / 발행년도:** {metadata['authors']} / {metadata['year']}\n")
            f.write(f"* **원문 링크:** [{source_url}]({source_url})\n")
            f.write(f"* **분석 모델:** {model_name}\n\n")
            f.write(f"#### 1. 📖 Abstract\n* **Original:** {metadata['abstract']}\n\n---\n")
        else:
            f.write(f"# 📄 [논문 분석: {paper_id}]\n\n* **원문 링크:** [{source_url}]({source_url})\n* **분석 모델:** {model_name}\n\n---\n")
        
        f.write(analysis_result)
        f.write("\n\n---\n#### 🖼️ 추출된 주요 그림(Figures)\n\n")
        
        if figure_data:
            for img_name, caption in figure_data:
                # 마크다운 태그로 이미지 삽입
                f.write(f"![{img_name}](images/{img_name})\n\n")
                # 그 아래 캡션 텍스트 그대로 작성
                f.write(f"> **{caption}**\n\n<br>\n\n") # 블록인용구 및 줄바꿈 적용
        else:
            f.write("문서에서 정식 Figure를 추출하지 못했습니다.\n")

    print(f"✅ 분석 및 Figure 추출 완료: {md_filename}")

if __name__ == "__main__":
    if os.path.exists("paper_links.txt"):
        with open("paper_links.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = [p.strip() for p in line.split(',')]
                url = parts[0]
                mode = parts[1].lower() if len(parts) > 1 and parts[1].lower() in ['pro', 'free'] else 'free'
                process_paper(url, mode)
                sleep(3)
