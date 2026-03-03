import os
import re
import urllib.request
import fitz  # PyMuPDF
from google import genai
from time import sleep

# --- [수정] 폴더 생성 로직 강화 ---
# 실행될 때마다 docs와 docs/images 폴더가 있는지 확인하고 없으면 만듭니다.
BASE_DIR = os.getcwd()
DOCS_DIR = os.path.join(BASE_DIR, "docs")
IMAGE_DIR = os.path.join(DOCS_DIR, "images")

os.makedirs(IMAGE_DIR, exist_ok=True)

# Gemini 설정
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = "당신은 AI/ML 논문 분석 전문가입니다. 논문을 한국어로 핵심 요약해주세요."

def get_latest_model(mode="free"):
    try:
        available_models = [m.name for m in client.models.list()]
        target = 'pro' if mode == 'pro' else 'flash'
        models = [m for m in available_models if 'gemini' in m and target in m and 'vision' not in m]
        models.sort(reverse=True)
        return models[0].replace('models/', '') if models else "gemini-2.0-flash"
    except: return "gemini-2.0-flash"

def extract_safe_images(pdf_path, paper_id):
    """그림 추출이 실패하더라도 최소 1개의 이미지는 무조건 생성하여 폴더 유실을 방지합니다."""
    doc = fitz.open(pdf_path)
    image_list = []
    
    # 1. [필살기] 1페이지를 통째로 캡처 (폴더 생성 보장용)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    snapshot_name = f"{paper_id}_main_page.png"
    pix.save(os.path.join(IMAGE_DIR, snapshot_name))
    image_list.append(snapshot_name)

    # 2. 개별 Figure 추출 시도
    for page_num, page in enumerate(doc):
        for img_idx, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                # 너무 작은 아이콘은 제외
                if pix.width < 100 or pix.height < 100: continue
                
                # 배경 흰색 보정
                rgb_pix = fitz.Pixmap(fitz.csRGB, pix.width, pix.height, 0)
                rgb_pix.clear_with_white()
                if pix.colorspace.n < 3: pix = fitz.Pixmap(fitz.csRGB, pix)
                rgb_pix.copy(pix, (0, 0, pix.width, pix.height))
                
                fname = f"{paper_id}_fig_{page_num+1}_{img_idx}.png"
                rgb_pix.save(os.path.join(IMAGE_DIR, fname))
                image_list.append(fname)
            except: continue
            
    doc.close()
    return image_list

def process_paper(url, mode="free"):
    # URL에서 ID 추출 (ArXiv 대응)
    match = re.search(r'arxiv\.org/(?:abs|pdf|html)/([0-9.]+)', url)
    paper_id = match.group(1).replace('.', '_') if match else "paper"
    md_path = os.path.join(DOCS_DIR, f"{paper_id}.md")

    # [테스트용] 기존 파일이 있어도 무조건 새로 분석하도록 설정
    # 실사용 시에는 아래 if문을 살려두면 좋습니다.
    # if os.path.exists(md_path): return

    pdf_path = f"temp_{paper_id}.pdf"
    req = urllib.request.Request(f"https://arxiv.org/pdf/{match.group(1)}.pdf" if match else url, 
                                 headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as resp, open(pdf_path, 'wb') as f:
            f.write(resp.read())
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return

    # 이미지 추출
    images = extract_safe_images(pdf_path, paper_id)
    
    # 본문 텍스트 추출 (앞 10페이지만)
    doc = fitz.open(pdf_path)
    text = "".join([p.get_text() for p in doc[:10]])
    doc.close()
    if os.path.exists(pdf_path): os.remove(pdf_path)

    model = get_latest_model(mode)
    try:
        res = client.models.generate_content(model=model, contents=SYSTEM_PROMPT + "\n\n" + text)
        analysis = res.text
    except Exception as e:
        print(f"API 에러: {e}")
        return

    # 마크다운 저장
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Analysis: {paper_id}\n\n{analysis}\n\n---\n### Figures & Snapshots\n\n")
        for img in images:
            # 웹사이트 경로에 맞게 images/파일명 형식으로 작성
            f.write(f"![{img}](images/{img})\n\n")
    
    print(f"✅ {paper_id} 분석 완료!")

if __name__ == "__main__":
    if os.path.exists("paper_links.txt"):
        with open("paper_links.txt", "r") as f:
            for line in f:
                if not line.strip(): continue
                parts = line.strip().split(',')
                process_paper(parts[0], parts[1].strip() if len(parts)>1 else "free")
                sleep(2)
