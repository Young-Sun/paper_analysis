import os
import re
import urllib.request
import fitz  # PyMuPDF
from google import genai
from time import sleep

# --- [1] 기본 경로 및 폴더 설정 ---
BASE_DIR = os.getcwd()
DOCS_DIR = os.path.join(BASE_DIR, "docs")
IMAGE_DIR = os.path.join(DOCS_DIR, "images")

# 부모 폴더까지 한 번에 생성 (exist_ok=True로 중복 생성 방지)
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- [2] API 클라이언트 설정 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """당신은 세계 최고 수준의 AI/ML 논문 분석 전문가입니다. 
제공된 논문 본문을 바탕으로 다음 구조에 맞춰 한국어로 핵심 내용을 요약해 주세요.

1. 🎯 연구 목적 (Why): 해결하고자 하는 문제와 배경
2. 🛠️ 연구 방법론 (How): 핵심 아이디어 및 모델 구조
3. 💡 주요 결과 (What): 실험 결과 및 성능 개선 지표
4. 🧠 핵심 인사이트: 이 연구가 갖는 의미와 실무 적용 아이디어
"""

def get_latest_model(mode="free"):
    """사용자 설정(free/pro)에 맞춰 최신 모델을 자동 탐색합니다."""
    try:
        available_models = [m.name for m in client.models.list()]
        target = 'pro' if mode == 'pro' else 'flash'
        models = [m for m in available_models if 'gemini' in m and target in m and 'vision' not in m]
        models.sort(reverse=True)
        return models[0].replace('models/', '') if models else "gemini-2.0-flash"
    except:
        return "gemini-2.0-flash"

def extract_cropped_figures(pdf_path, paper_id):
    """
    논문 페이지 내 'Fig' 텍스트를 찾아 주변 영역을 고화질로 잘라냅니다.
    배경색 오류가 없으며 그림과 캡션을 함께 저장합니다.
    """
    doc = fitz.open(pdf_path)
    figure_data = []
    
    for p_idx, page in enumerate(doc):
        # 1. 페이지 내 텍스트 블록 탐색
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            # 'Fig 1', 'Figure 2' 등의 패턴 탐색
            match = re.search(r'^\s*(Fig|Figure|FIG)\.?\s*(\d+)', text, re.IGNORECASE)
            
            if match:
                fig_num = match.group(2)
                # 캡션의 좌표 (x0, y0, x1, y1)
                caption_rect = fitz.Rect(b[:4])
                
                # 2. 크롭 영역 계산 (캡션 위 400pt부터 캡션 아래 20pt까지)
                # 대부분의 논문은 그림 아래에 캡션이 있으므로 위쪽을 넓게 잡습니다.
                crop_rect = fitz.Rect(0, caption_rect.y0 - 450, page.rect.width, caption_rect.y1 + 30)
                
                # 페이지 경계를 넘지 않도록 조정
                crop_rect.intersect(page.rect)
                
                # 3. 고화질 렌더링 (2.0배 확대)
                # clip 옵션으로 특정 영역만 사진 찍듯 가져옵니다.
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop_rect)
                
                img_name = f"{paper_id}_Fig{fig_num}_p{p_idx+1}.png"
                pix.save(os.path.join(IMAGE_DIR, img_name))
                figure_data.append((img_name, text))
                
    doc.close()
    return figure_data

def process_paper(url, mode="free"):
    # ArXiv ID 추출
    match = re.search(r'(\d+\.\d+)', url)
    if not match:
        print(f"URL 형식이 올바르지 않습니다: {url}")
        return
    
    aid = match.group(1)
    aid_clean = aid.replace('.', '_')
    md_path = os.path.join(DOCS_DIR, f"{aid_clean}.md")

    # [중요] 이미 분석된 논문은 스킵 (테스트 시에는 이 줄을 주석 처리 하세요)
    if os.path.exists(md_path):
        print(f"이미 분석된 논문입니다: {aid}")
        return

    print(f"\n[{aid}] 분석 시작...")
    
    # 1. PDF 다운로드
    pdf_path = f"temp_{aid_clean}.pdf"
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(f"https://arxiv.org/pdf/{aid}.pdf", headers=headers)
    
    try:
        with urllib.request.urlopen(req) as resp, open(pdf_path, 'wb') as f:
            f.write(resp.read())
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return

    # 2. 영역 크롭 방식으로 그림 추출
    print(" -> 그림 및 캡션 영역 추출 중...")
    figures = extract_cropped_figures(pdf_path, aid_clean)
    
    # 3. 텍스트 추출 및 AI 분석
    doc = fitz.open(pdf_path)
    # 토큰 절약을 위해 앞쪽 10페이지만 텍스트 추출
    text_content = "".join([p.get_text() for p in doc[:10]])
    doc.close()
    
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    print(" -> Gemini AI 분석 중...")
    model_name = get_latest_model(mode)
    try:
        response = client.models.generate_content(
            model=model_name, 
            contents=SYSTEM_PROMPT + "\n\n--- PAPER TEXT ---\n" + text_content
        )
        analysis_result = response.text
    except Exception as e:
        print(f"API 호출 에러: {e}")
        return

    # 4. 마크다운 보고서 조립 및 저장
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 📄 논문 분석: {aid}\n\n")
        f.write(f"* **원문 링크:** [{url}]({url})\n")
        f.write(f"* **사용 모델:** {model_name}\n\n---\n")
        f.write(analysis_result)
        f.write("\n\n---\n### 🖼️ 주요 그림 및 표 (Figures)\n\n")
        
        if figures:
            for img, cap in figures:
                f.write(f"![fig](images/{img})\n\n")
                f.write(f"> **{cap}**\n\n<br>\n\n")
        else:
            f.write("문서에서 정식 Figure를 감지하지 못했습니다.\n")
            
    print(f"✅ 분석 완료: {md_path}")

if __name__ == "__main__":
    links_file = "paper_links.txt"
    if os.path.exists(links_file):
        with open(links_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                parts = line.split(',')
                url = parts[0].strip()
                mode = parts[1].strip().lower() if len(parts) > 1 else "free"
                
                process_paper(url, mode)
                sleep(3) # API 할당량 보호
