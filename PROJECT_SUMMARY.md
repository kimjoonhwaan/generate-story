# RAG Story Generator — 프로젝트 종합 정리

## 1) 프로젝트 개요
- **목표**: 업로드한 문서의 어휘(RAG)를 활용해 키워드 기반 영어 스토리를 생성
- **핵심**: 벡터 검색(Chroma) → 컨텍스트 주입 → GPT-4/4o 스토리 생성 → RAG 어휘 제약 및 품질 검사
- **UI**: Streamlit 웹앱(업로드/설정/생성/어휘 뷰어/로그/통계)

## 2) 실행/모드
- 기본 모드: 벡터 검색 → OpenAI 직접 호출(재시도/품질검사 내장)
- LangChain 모드: ChatPromptTemplate + ChatOpenAI/AzureChatOpenAI + 파서(단일 패스)
- 멀티에이전트 모드: LangGraph + ReAct(도구 호출)로 "검색→생성→평가→수정" 오케스트레이션, 단계 로그 제공

설정 옵션(UI 사이드바)
- Use OpenAI/Azure OpenAI API
- Use only RAG vocabulary(등록 어휘 + 필수 문법어만 허용)
- Use LangChain pipeline
- Use Multi‑Agent (LangGraph + ReAct)

## 3) 아키텍처/구성요소
- `app.py`: Streamlit UI, 파일 업로드/설정/결과 표시, 멀티에이전트 로그 표시
- `rag_system.py`: 검색 → 생성 오케스트레이션
- `story_generator.py`: 스토리 생성(길이 설정, 프롬프트, 재시도/품질검사, RAG 어휘 분석)
- `text_processor.py`: TXT/CSV/PDF 읽기, 정제/청크/키워드 추출(PDF 분석 지원)
- `vector_db.py`: Chroma 영구 클라이언트, 문서 추가/검색, 어휘 관리
- `lc_pipeline.py`: LangChain 체인(프롬프트 템플릿 → LLM → 파서)
- `agents/agent_flow.py`: LangGraph + ReAct 멀티에이전트(도구, 상태, 로그)
- `requirements.txt`: 의존성
- `DEPLOYMENT.md`: 배포 가이드
- `PROJECT_STRUCTURE.md`: 파일 구조/프롬프트 문서

## 4) 임베딩 전략(통일)
- **우선순위**: OpenAIEmbeddings → 실패 시 AzureOpenAIEmbeddings(배포명/버전 필요)
- **컬렉션 분리**: 차원 충돌 방지(`rag_documents_openai`, `rag_documents_aoai` 등)
- **빈 입력 가드**: 임베딩 실패 시 즉시 예외, 빈 `query_embeddings` 방지
- Azure 400($.input) 대응: 입력 정제, 길이 제한, 배치 실패 시 단건 재시도

Azure 환경변수(예)
```
AOAI_API_KEY=...
AOAI_ENDPOINT=https://<리소스>.openai.azure.com
AOAI_API_VERSION=2024-06-01
AOAI_DEPLOY_GPT4O=<gpt-4o 배포명>
AOAI_EMBEDDING_DEPLOYMENT=<text-embedding 배포명>
```

## 5) RAG 어휘 제약/분석
- 허용 어휘: 등록 어휘 + 필수 문법어(articles, be/aux, pronouns, prepositions, conjunctions 등)
- 품질 검사(기본 경로):
  - 최소 길이(단어 수)
  - 반복 단어(thing/something) 과다 차단
  - RAG 어휘 사용률(기본 40% 이상)
- 분석 출력: 스토리 하단 사용률, 비‑RAG 단어 목록(간단 품사 추정)
- 멀티에이전트: RAG-only일 때 허용 어휘가 적으면 자동으로 전체 어휘로 보강

## 6) PDF/텍스트 처리
- PDF: `pdfplumber` 우선, `PyPDF2` 폴백(페이지별 추출/통계 출력 가능)
- 어휘 추출: 순수 영문/하이픈/아포스트로피/숫자접두/대문자 약어 포함
- 청크: 문장 누적 기반, `chunk_overlap`로 경계 손실 완화

## 7) 멀티에이전트(선택)
- LangGraph 상태 머신: Retrieve → Generate → Evaluate → Revise(최대 2회)
- ReAct 도구: `generate_draft`, `vocab_analysis` (JSON 단일 입력, 기본값 바인딩)
- 프롬프트: 엄격한 ReAct 포맷(Thought/Action/Action Input/Observation/Final Answer)
- 로그: 콘솔(`[MultiAgent] ...`) + UI(Agent Progress Logs) 표시
- 속도: 단일 패스 대비 느림(여러 번 LLM 호출 및 파싱 재시도). 필요 시 기본/LC 모드 권장

## 8) 주요 이슈/해결
- NumPy 2.0 호환성 → `numpy==1.26.4` 고정
- huggingface-hub 충돌 → `>=0.34,<1.0` 상향
- Azure 임베딩 400($.input) → 입력 정제/단건 재시도/배포명·버전 확인
- `query_embeddings` 없음 → 임베딩 실패 시 예외 던지도록 수정
- 멀티에이전트 빈 쿼리 → 툴 의존 제거, 첫 키워드 직접 검색, 기본값 바인딩
- 멀티에이전트 RAG-only 실패 → 허용 어휘 자동 보강(최소 임계)
- GitHub Push 차단(API 키) → `.env` 제거, `.gitignore` 추가, 히스토리 재작성

## 9) 실행 방법(요약)
```bash
# 의존성
python -m pip install -r requirements.txt

# (선택) sentence-transformers 설치 권장
python -m pip install 'sentence-transformers>=2.3.0' 'huggingface-hub>=0.34,<1.0'

# 앱 실행
python main.py --web
```

## 10) 사용 가이드
- 문서 업로드(TXT/CSV/PDF) → 키워드 입력(쉼표 구분) → 모드/옵션 선택 → Generate
- 모드 선택 가이드
  - **기본**: 가장 빠름/안정, 일상적 사용 권장
  - **LangChain**: 프롬프트 관리/확장 용이, 속도 유사
  - **멀티에이전트**: 품질 루프/로그 필요 시. 느림 주의
- RAG-only: 등록 어휘가 충분하지 않으면 사용률/자연스러움 저하 가능 → 어휘 보강 권장

## 11) 향후 개선
- Fast Mode(멀티에이전트 단일 패스/수정루프 비활성)
- 난이도/스타일 템플릿(초/중/고급)
- 캐싱·타임아웃·경량 모델 도입으로 지연 단축
- API(FastAPI) 제공 및 배치 처리

---
본 문서는 현재 저장소 내 구현/설정을 기준으로 최신 상태를 요약합니다. 더 자세한 내용은 `README.md`, `DEPLOYMENT.md`, `PROJECT_STRUCTURE.md`를 참고하세요. 