# 📘 Abstractive Summarizer  
HuggingFace Transformers 기반 영어 뉴스 기사 요약 오픈소스 SW

---

## 1. 프로젝트 개요 (Project Overview)
이 프로젝트는 **긴 영어 뉴스 기사**를 입력하면 핵심 내용을 자동으로 추출하여  
**요약 텍스트를 생성하는 데모 소프트웨어**입니다.

HuggingFace의 `facebook/bart-large-cnn` 모델을 사용해  
간단한 Python 스크립트 실행만으로 요약 결과를 확인할 수 있습니다.

---

## 2. 데모 예시 (Demo Example)

### ✔ 입력(article) 예시
<img width="1483" height="435" alt="image" src="https://github.com/user-attachments/assets/6ff5b5fe-a671-4f13-b25a-6e874c7eff34" />
A volcano in Ethiopia has erupted for the first known time in 10,000 years, spewing plumes of thick smoke and ash high into the sky and impacting air travel thousands of miles away in India.
The long-dormant Hayli Gubbi volcano in the Afar region in Ethiopia’s northeast roared to life Sunday, covering the neighboring villages in dust and creating challenges for farmers.
While no casualties were reported, the eruption poses a threat to the local community of livestock herders by smothering vital grazing lands, local administrator Mohammed Seid told The Associated Press
Residents described hearing a terrifying blast at the moment of the eruption.
“It felt like a sudden bomb had been thrown with smoke and ash,” local resident Ahmed Abdela told the news agency.
The eruption was visible from satellites, with NASA images showing thick plumes of dust rising into the sky and billowing across the Red Sea.
Volcanic clouds from the eruption drifted over Yemen, Oman, and into Pakistan and India, according to the Toulouse Volcanic Ash Advisory Center.
Pakistan’s Meteorological Department issued a warning after ash entered its airspace late on Monday.
In India, flag carrier Air India cancelled several domestic and international flights to carry out “precautionary checks on those aircraft which had flown over certain geographical locations after the Hayli Gubbi volcanic eruption,” it said on X.
Delhi, which is experiencing a wave of severe air pollution, is not expected to be significantly affected because the ash is drifting at a high altitude, India’s Meteorological Department (IMD) said.
The plumes are expected to rapidly move eastwards, the IMD added.
Located about 800 kilometers (500 miles) northeast of capital Addi Ababa, Hayli Gubbi is the southernmost volcano of the Erta Ale Range, a volcanic chain in Ethiopia’s Afar region.
It rises about 500 meters in altitude and sits within a zone of intense geological activity where two tectonic plates meet.

### ✔ 요약 결과(summary)
<img width="1681" height="117" alt="image" src="https://github.com/user-attachments/assets/48837a6f-1d63-48f6-9b4d-83808139be0d" />
The long-dormant Hayli Gubbi volcano in the Afar region in Ethiopia’s northeast roared to life Sunday. The eruption was visible from satellites, with NASA images showing thick plumes of dust rising into the sky. Volcanic clouds from the eruption drifted over Yemen, Oman, and into Pakistan and India.

---

## 3. 사용한 패키지 / 버전 (Dependencies)

### ✔ requirements.txt
transformers==4.44.0
torch
sentencepiece

---

## 4. 실행 방법 (How to Run)

### ✔ 가상환경 생성
python -m venv venv

### ✔ 가상환경 활성화
venv\Scripts\activate

### ✔ 패키지 설치
pip install -r requirements.txt

### ✔ 데모 실행
python test_summary.py

---

## 5. 디렉토리 구조 (Directory Layout)
abstractive-summarizer/
│
├── src/
│   └── summarizer.py   # 요약 모델 로직
│
├── test_summary.py     # 데모 실행 파일
│
├── requirements.txt    # 필요한 패키지 목록
│
└── README.md           # 프로젝트 설명 문서

---

## 6. 소스 코드 설명 (Source Code Explanation)

### # summarizer.py
- HuggingFace BART summarization 모델 로드  
- AutoTokenizer로 텍스트 전처리  
- `summarize()` 함수에서 입력 텍스트 → 요약 텍스트 생성  

### # test_summary.py
- 기사 원문(article) 정의  
- summarize() 호출  
- 요약 결과 출력  

---

## 7. 참고 자료 (References)

### # HuggingFace Model
- https://huggingface.co/facebook/bart-large-cnn  

### # Transformers Documentation
- https://huggingface.co/docs/transformers/index  

---

## 8. 팀 협업 방식 (Team Collaboration)

### # GitHub 브랜치 규칙
- 모든 주요 파일은 **main 브랜치** 기준으로 확인 가능해야 함  
- 기능 개발 후 **Pull Request 제출 → 팀 리더 승인 후 merge**  
- commit history에 작업 이력이 남도록 진행  

### # 팀원 작업 방식
- 각 팀원이 local에서 개발 후 `git push`  
- 팀 리더가 merge만 담당하여 작업 이력 명확히 관리  

---

## 9. 프로젝트 특징 요약 (Highlights)

### # 핵심 기능
- HuggingFace 기반 **Abstractive Summarization**
- 영어 뉴스 기사 요약 자동 생성

### # 장점
- 간단한 Y/N 설치 및 실행 구조  
- 오픈소스로 확장 가능  
- 코드 구조 명확하여 유지보수 용이  

---

