import os
import re
import urllib.request
import fitz  # PyMuPDF
from google import genai
from time import sleep

# --- [1] 경로 및 폴더 설정 ---
BASE_DIR = os.getcwd()
DOCS_DIR = os.path.join(BASE_DIR, "docs")
IMAGE_DIR = os.path.join(DOCS_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- [2] API 설정 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """당신은 AI/ML 논문 분석 전문가입니다. 
다음 구조에 맞춰 한국어로 핵심 요약해 주세요.
1. 🎯 연구 목적  2. 🛠️ 연구 방법론  3. 💡 주요 결과  4. 🧠 핵심 인사이트"""

def get_latest_model(mode="free"):
    try:
        available_models = [m.name for m in client.models.list()]
        target = 'pro' if mode == 'pro' else 'flash'
        models = [m for m in available_models if 'gemini' in m and target in m and 'vision' not in m]
        models.sort(reverse=True)
        return models[0].replace('models/', '') if models else "gemini-2.0-flash"
    except: return "gemini-2.0-flash"

def extract_images_with_white_bg(pdf_path, paper_id):
    """PyMuPDF를 사용하여 이미지를 추출하고, 투명 배경을 흰색으로 강제 보정합니다."""
    doc = fitz.open(pdf_path)
    image_list = []
    
    for page_num, page in enumerate(doc):
        # 페이지 내 이미지 목록 가져오기
        page_imgs = page.get_images(full=True)
        
        for img_idx, img in enumerate(page_imgs):
            xref = img[0]
            try:
                # 1. 원본 이미지 픽스맵 추출
                pix = fitz.Pixmap(doc, xref)
                
                # 너무 작은 이미지(로고, 아이콘 등) 필터링 (가로 세로 120px 미만)
                if pix.width < 120 or pix.height < 120:
                    continue

                # 2. [배경 보정 핵심] 흰색 배경 도화지 생성
                # pix.alpha가 있든 없든, RGB 모드로 흰색 도화지를 깔고 그 위에 얹습니다.
                # 이렇게 하면 투명한 부분이 검게 변하는 현상이 100% 해결됩니다.
                rgb_pix = fitz.Pixmap(fitz.csRGB, pix.width, pix.height, 0)
                rgb_pix.clear_with_white()
                
                # 원본이 Gray나 CMYK일 경우 RGB로 변환하여 복사
                if pix.colorspace.n < 3:
                    temp_pix = fitz.Pixmap(fitz.csRGB, pix)
                    rgb_pix.copy(temp_pix, (0, 0, pix.width, pix.height))
                else:
                    rgb_pix.copy(pix, (0, 0, pix.width, pix.height))
                
                # 3. 파일 저장
                img_name = f"{paper_id}_p{page_num+1}_{img_idx}.png"
                rgb_pix.save(os.path.join(IMAGE_DIR, img_name))
                image_list.append(img_name)
                
                pix = None
                rgb_pix = None
            except Exception as e:
                print(f"이미지 추출 오류 (xref: {xref}): {e}")
                continue
                
    doc.close()
    return image_list

def process_paper(url, mode="free"):
    # ArXiv ID 추출
    match = re.search(r'(\d+\.\d+)', url)
    if not match: return
    
    aid = match.group(1)
    aid_clean = aid.replace('.', '_')
    md_path = os.path.join(DOCS_DIR, f"{aid_clean}.md")

    # 기존 분석 스킵 (새로 하시려면 이 줄을 잠시 주석처리 하세요)
    if os.path.exists(md_path): return

    print(f"[{aid}] 분석 및 이미지 추출 시작...")
    
    # PDF 다운로드
    pdf_path = f"temp_{aid_clean}.pdf"
    req = urllib.request.Request(f"https://arxiv.org/pdf/{aid}.pdf", headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp, open(pdf_path, 'wb') as f:
            f.write(resp.read())
    except: return

    # 이미지 추출 (배경 보정 적용)
    images = extract_images_with_white_bg(pdf_path, aid_clean)
    
    # 텍스트 분석
    doc = fitz.open(pdf_path)
    text = "".join([p.get_text() for p in doc[:10]])
    doc.close()
    if os.path.exists(pdf_path): os.remove(pdf_path)

    model_name = get_latest_model(mode)
    try:
        res = client.models.generate_content(model=model_name, contents=SYSTEM_PROMPT + "\n\n" + text)
        analysis = res.text
    except: return

    # 마크다운 저장
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Analysis {aid}\n\n* **원문:** {url}\n\n{analysis}\n\n---\n### Figures\n\n")
        for img in images:
            # 캡션은 수동 수정을 위해 이미지 파일명만 간단히 기재합니다.
            f.write(f"![{img}](images/{img})\n\n")
            
    print(f"✅ 완료: {md_path}")

if __name__ == "__main__":
    if os.path.exists("paper_links.txt"):
        with open("paper_links.txt", "r") as f:
            for line in f:
                if not line.strip(): continue
                parts = line.split(',')
                process_paper(parts[0].strip(), parts[1].strip().lower() if len(parts)>1 else "free")
                sleep(3)
