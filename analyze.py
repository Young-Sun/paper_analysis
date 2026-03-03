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

# 시스템 프롬프트 (기존 유지)
SYSTEM_PROMPT = """...기존 내용..."""

def get_latest_model(mode="free"):
    try:
        available_models = [m.name for m in client.models.list()]
        target = 'pro' if mode == 'pro' else 'flash'
        models = [m for m in available_models if 'gemini' in m and target in m and 'vision' not in m]
        models.sort(reverse=True)
        return models[0].replace('models/', '') if models else "gemini-1.5-flash"
    except: return "gemini-1.5-flash"

def extract_all_images(pdf_path, paper_id):
    """모든 이미지를 흰색 배경으로 보정하여 추출합니다."""
    doc = fitz.open(pdf_path)
    image_list = []
    
    for page_num, page in enumerate(doc):
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            
            # 너무 작은 아이콘(10KB 이하)은 무시
            if len(base_image["image"]) < 10000: continue
            
            filename = f"{paper_id}_p{page_num+1}_{img_idx}.png"
            filepath = os.path.join(IMAGE_DIR, filename)
            
            try:
                pix = fitz.Pixmap(doc, xref)
                # 배경색 보정 (흰색 도화지 위에 그리기)
                rgb_pix = fitz.Pixmap(fitz.csRGB, pix.width, pix.height, 0)
                rgb_pix.clear_with_white()
                
                if pix.colorspace.n < 3:
                    temp = fitz.Pixmap(fitz.csRGB, pix)
                    rgb_pix.copy(temp, (0, 0, pix.width, pix.height))
                else:
                    rgb_pix.copy(pix, (0, 0, pix.width, pix.height))
                
                rgb_pix.save(filepath)
                image_list.append(filename)
            except: continue
            
    doc.close()
    return image_list

def process_paper(url, mode="free"):
    # ArXiv ID 추출 및 파일명 설정
    match = re.search(r'arxiv\.org/(?:abs|pdf|html)/([0-9.]+)', url)
    paper_id = match.group(1).replace('.', '_') if match else "paper"
    md_path = os.path.join(DOCS_DIR, f"{paper_id}.md")

    # 기존 파일 삭제 (강제 재분석을 위해 테스트 중에는 체크 해제 가능)
    # if os.path.exists(md_path): return

    pdf_path = f"temp_{paper_id}.pdf"
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(f"https://arxiv.org/pdf/{match.group(1)}.pdf" if match else url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as resp, open(pdf_path, 'wb') as f:
            f.write(resp.read())
    except: return

    # 1. 이미지 추출 (단순/확실한 방식)
    images = extract_all_images(pdf_path, paper_id)
    
    # 2. 텍스트 분석
    doc = fitz.open(pdf_path)
    text = "".join([p.get_text() for p in doc])
    doc.close()
    os.remove(pdf_path)

    model = get_latest_model(mode)
    try:
        res = client.models.generate_content(model=model, contents=SYSTEM_PROMPT + "\n\n" + text)
        analysis = res.text
    except: return

    # 3. 마크다운 저장
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Analysis: {paper_id}\n\n{analysis}\n\n---\n### Figures\n\n")
        for img in images:
            f.write(f"![{img}](images/{img})\n\n")

if __name__ == "__main__":
    if os.path.exists("paper_links.txt"):
        with open("paper_links.txt", "r") as f:
            for line in f:
                if not line.strip(): continue
                parts = line.strip().split(',')
                process_paper(parts[0], parts[1].strip() if len(parts)>1 else "free")
                sleep(2)
