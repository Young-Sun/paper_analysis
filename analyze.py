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

# 시스템 프롬프트 (생략 - 기존과 동일하게 유지)
SYSTEM_PROMPT = """...기존 내용 유지..."""

def get_latest_model(mode="free"):
    try:
        available_models = [m.name for m in client.models.list()]
        target_models = [m for m in available_models if 'gemini' in m and ('pro' if mode == 'pro' else 'flash') in m and 'vision' not in m]
        target_models.sort(reverse=True)
        return target_models[0].replace('models/', '') if target_models else ("gemini-2.0-flash" if mode == "free" else "gemini-2.0-pro")
    except: return "gemini-1.5-flash"

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
        return {"title": entry.find('atom:title', ns).text.replace('\n', ' ').strip(), "abstract": entry.find('atom:summary', ns).text.replace('\n', ' ').strip(), "authors": ", ".join([a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]), "year": entry.find('atom:published', ns).text[:4]}
    except: return None

def download_pdf_safely(url: str, output_path: str):
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as f: f.write(response.read())
        return True
    except: return False

def extract_figures_and_captions(pdf_path: str, paper_id: str):
    doc = fitz.open(pdf_path)
    figure_data = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        # 1. 페이지 내 모든 이미지 정보 가져오기
        page_imgs = page.get_images(full=True)
        
        for block in blocks:
            text = block[4].strip()
            # Figure 캡션 탐색 (Fig 1, Figure 1 등)
            match = re.search(r'^\s*(Fig|Figure|FIG|FIGURE)\.?\s*(\d+)', text, re.IGNORECASE)
            
            if match:
                fig_num = match.group(2)
                caption_text = text.replace('\n', ' ')
                b_rect = fitz.Rect(block[:4])
                
                # 캡션 주변(위/아래 300포인트) 영역에서 가장 큰 이미지 찾기
                search_area = fitz.Rect(b_rect.x0 - 50, b_rect.y0 - 400, b_rect.x1 + 50, b_rect.y1 + 400)
                
                best_xref = None
                max_size = 0
                
                for img in page_imgs:
                    xref = img[0]
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        i_rect = img_rects[0]
                        # 검색 영역 내에 있고, 일정 크기 이상인 것 중 가장 큰 것 선택
                        if search_area.intersects(i_rect):
                            size = i_rect.width * i_rect.height
                            if size > max_size and size > 5000: # 너무 작은 조각 무시
                                max_size = size
                                best_xref = xref
                
                if best_xref:
                    img_name = f"{paper_id}_Fig{fig_num}.png"
                    img_path = os.path.join(IMAGE_DIR, img_name)
                    
                    try:
                        # 배경색 강제 보정 (투명/검정 -> 흰색)
                        pix = fitz.Pixmap(doc, best_xref)
                        
                        # RGB로 변환하여 투명도 제거 및 흰색 배경 합성
                        if pix.alpha or pix.colorspace.n < 3:
                            new_pix = fitz.Pixmap(fitz.csRGB, pix.width, pix.height, 0)
                            new_pix.clear_with_white()
                            # 겹쳐 그리기
                            new_pix.copy(pix, (0, 0, pix.width, pix.height))
                            pix = new_pix
                        
                        pix.save(img_path)
                        figure_data.append((img_name, caption_text))
                    except: pass
    doc.close()
    return figure_data

def process_paper(source_url: str, mode: str = "free"):
    arxiv_id = parse_arxiv_id(source_url)
    paper_id = arxiv_id.replace('.', '_') if arxiv_id else "paper"
    md_filename = os.path.join(DOCS_DIR, f"{paper_id}.md")
    
    if os.path.exists(md_filename): return

    model_name = get_latest_model(mode)
    pdf_path = f"temp_{paper_id}.pdf"
    
    if not download_pdf_safely(f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else source_url, pdf_path): return

    # 분석 및 데이터 추출
    figure_data = extract_figures_and_captions(pdf_path, paper_id)
    
    # 본문 텍스트 추출 (References 제외)
    doc = fitz.open(pdf_path)
    text = "".join([p.get_text() for p in doc])
    ref_match = re.search(r'\n\s*(References|Bibliography|REFERENCES)\s*\n', text)
    body_text = text[:ref_match.start()] if ref_match else text
    doc.close()
    os.remove(pdf_path)

    # API 호출
    try:
        res = client.models.generate_content(model=model_name, contents=SYSTEM_PROMPT + "\n\n" + body_text)
        analysis = res.text
    except: return

    # 마크다운 파일 작성
    metadata = fetch_arxiv_metadata(arxiv_id) if arxiv_id else None
    with open(md_filename, "w", encoding="utf-8") as f:
        if metadata:
            f.write(f"# 📄 {metadata['title']}\n\n* **원문:** {source_url}\n\n{analysis}\n\n---\n### 🖼️ Figures\n\n")
        
        for name, cap in figure_data:
            f.write(f"![{name}](images/{name})\n\n> **{cap}**\n\n")

if __name__ == "__main__":
    if os.path.exists("paper_links.txt"):
        with open("paper_links.txt", "r") as f:
            for line in f:
                if not line.strip(): continue
                p = [i.strip() for i in line.split(',')]
                process_paper(p[0], p[1].lower() if len(p)>1 else 'free')
                sleep(2)
