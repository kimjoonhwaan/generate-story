# RAG Story Generator - 프로젝트 구조 및 프롬프트 문서

## 📁 프로젝트 구조

```
projectRag2/
├── 📄 app.py                    # Streamlit 웹 인터페이스
├── 📄 main.py                   # 메인 실행 파일
├── 📄 rag_system.py             # RAG 시스템 오케스트레이터
├── 📄 vector_db.py              # ChromaDB 벡터 데이터베이스 관리
├── 📄 text_processor.py         # 텍스트 처리 및 파일 읽기
├── 📄 story_generator.py        # 스토리 생성 엔진 (핵심)
├── 📄 requirements.txt          # Python 의존성
├── 📄 README.md                 # 프로젝트 개요
├── 📄 env_template.txt          # 환경변수 템플릿
├── 📄 sample_stories.txt        # 샘플 텍스트 파일
├── 📄 test_rag.py              # 테스트 스크립트
└── 📄 PROJECT_STRUCTURE.md      # 이 문서
```

## 🏗️ 아키텍처 개요

### 핵심 컴포넌트

1. **Vector Database (ChromaDB)**
   - 문서 임베딩 저장
   - 의미적 검색 수행
   - RAG 어휘 관리

2. **Text Processor**
   - 다중 파일 형식 지원 (TXT, CSV, PDF)
   - 텍스트 청킹 및 정제
   - 키워드 추출

3. **Story Generator**
   - OpenAI GPT-4 기반 생성
   - 어휘 제한 기능
   - 품질 검사 및 재시도

4. **Web Interface (Streamlit)**
   - 사용자 친화적 UI
   - 실시간 피드백
   - 어휘 분석 표시

## 📄 핵심 파일 상세 분석

### 1. `story_generator.py` - 스토리 생성 엔진

#### 주요 클래스: `StoryGenerator`

```python
class StoryGenerator:
    def __init__(self, use_openai: bool = True)
    def generate_story_with_openai(self, keywords, context_documents, story_length, available_vocabulary)
    def generate_story_locally(self, keywords, context_documents, story_length, available_vocabulary)
    def generate_story(self, keywords, context_documents, story_length, use_vocabulary_restriction, available_vocabulary)
    def _annotate_non_rag_words(self, story, available_vocabulary)
```

#### 핵심 기능

1. **OpenAI 기반 생성**
   - GPT-4 모델 사용
   - 엄격한 어휘 제한
   - 품질 검사 및 재시도

2. **로컬 템플릿 생성**
   - OpenAI 실패 시 백업
   - 기본 문장 패턴 사용

3. **어휘 분석**
   - RAG 외 단어 식별
   - 카테고리별 분류
   - 사용률 통계

### 2. `vector_db.py` - 벡터 데이터베이스

#### 주요 클래스: `VectorDB`

```python
class VectorDB:
    def __init__(self, collection_name: str = "rag_documents")
    def add_documents(self, texts: List[str], metadatas: List[Dict])
    def search(self, query: str, n_results: int = 5)
    def get_vocabulary(self) -> Set[str]
    def get_filtered_vocabulary(self, keywords: str, context_documents: List[str])
    def clear_collection(self)
```

#### 핵심 기능

1. **임베딩 관리**
   - SentenceTransformer 사용
   - TF-IDF 백업 시스템
   - 코사인 유사도 검색

2. **어휘 관리**
   - 문서별 단어 추출
   - 필터링 및 정제
   - 영구 저장

### 3. `text_processor.py` - 텍스트 처리

#### 주요 클래스: `TextProcessor`

```python
class TextProcessor:
    def read_file(self, file_path: str) -> str
    def clean_text(self, text: str) -> str
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]
    def extract_keywords(self, text: str) -> List[str]
    def process_file(self, file_path: str) -> Tuple[List[str], Dict]
```

#### 지원 파일 형식

1. **TXT 파일**
   - 기본 텍스트 읽기
   - UTF-8 인코딩 지원

2. **CSV 파일**
   - pandas 기반 처리
   - 컬럼별 텍스트 추출

3. **PDF 파일**
   - pdfplumber 우선 사용
   - PyPDF2 백업
   - 교육용 교재 형식 지원

### 4. `rag_system.py` - 시스템 오케스트레이터

#### 주요 클래스: `RAGSystem`

```python
class RAGSystem:
    def __init__(self, use_openai: bool = True)
    def add_file_to_database(self, file_path: str) -> bool
    def search_and_generate_story(self, keywords, story_length, n_results, use_only_rag_vocabulary)
    def get_database_stats(self) -> Dict
    def clear_database(self)
```

## 🤖 프롬프트 엔지니어링

### 1. OpenAI 스토리 생성 프롬프트

#### 시스템 메시지
```
"You are a story writer who MUST follow vocabulary restrictions EXACTLY. 
When given a vocabulary list, you can ONLY use words from that list plus basic grammar words (a, an, the, is, are, was, were, etc.). 
If you cannot express something with the allowed words, you MUST rephrase or find alternatives from the vocabulary list. 
This is a strict vocabulary exercise."
```

#### 사용자 프롬프트 구조

```python
prompt = f"""Create an engaging English story using these requirements:

**PRIMARY KEYWORDS (MUST include all):** {primary_keywords}
**SECONDARY KEYWORDS (include if possible):** {secondary_keywords}

**STORY LENGTH:** {settings['sentences']} ({settings['words']})

**CONTEXT INFORMATION:**
{context_text}

{vocabulary_instruction}

**STORY REQUIREMENTS:**
- Write EXACTLY {settings['sentences']} with approximately {settings['words']}
- Create a complete story with clear beginning, middle, and end
- FOCUS heavily on the primary keywords - they should be central to the plot
- Use vivid descriptions and engaging narrative
- CRITICAL: Every sentence must be grammatically perfect and meaningful
- Each sentence must make logical sense
- Use proper punctuation and capitalization
- Connect ideas smoothly with conjunctions
- Ensure the story flows naturally from sentence to sentence

**EXAMPLE STRUCTURE:**
- Opening: Introduce main character and setting using primary keywords
- Development: Build conflict/challenge involving the keywords  
- Resolution: Resolve the situation meaningfully

Write the story now:"""
```

#### 어휘 제한 프롬프트

```python
vocabulary_instruction = f"""
**CRITICAL VOCABULARY RESTRICTION - YOU MUST FOLLOW THIS EXACTLY:**

**ONLY USE THESE WORDS:**
1. RAG Vocabulary: {vocab_sample}
2. Essential Grammar Words: a, an, the, is, are, was, were, be, have, has, had, do, does, did, will, would, can, could, should, may, might, must, shall, and, or, but, so, if, when, where, what, who, how, why, in, on, at, by, for, with, to, from, about, I, you, he, she, it, we, they, me, him, her, us, them, my, your, his, her, its, our, their, this, that, these, those, here, there, now, then

**STRICT RULES:**
- DO NOT use ANY words outside these two lists
- If you need a word not in the lists, find an alternative from the RAG vocabulary
- Rephrase sentences to use only allowed words
- Every single word must be from the allowed lists
- This is a vocabulary exercise - creativity within constraints

**EXAMPLE:**
❌ WRONG: "In the heart of a bustling city" (heart, bustling, city not in lists)
✅ CORRECT: "In the center of a busy town" (using allowed words)

**PRIORITY: Use RAG vocabulary words as much as possible!**
"""
```

### 2. 품질 검사 및 재시도 로직

#### 품질 검사 기준

```python
# 1. 길이 검사
if len(story.split()) < 50:
    continue  # 재시도

# 2. 반복 단어 검사
if words.count('thing') > len(words) * 0.1 or words.count('something') > len(words) * 0.1:
    continue  # 재시도

# 3. RAG 어휘 사용률 검사
if rag_usage_rate < 0.4:  # 40% 미만
    continue  # 재시도
```

#### 재시도 메커니즘

```python
max_retries = 3
best_story = None
best_rag_rate = 0

for attempt in range(max_retries):
    # OpenAI API 호출
    # 품질 검사
    # 최고 품질 스토리 저장
    
# 실패 시 최고 품질 스토리 반환
if best_story:
    return best_story
```

## 🔧 설정 및 환경변수

### 필수 환경변수

```bash
# .env 파일
OPENAI_API_KEY=your_openai_api_key_here
```

### 주요 설정값

```python
# 길이 설정
length_settings = {
    "short": {"tokens": 300, "sentences": "3-5 sentences", "words": "100-200 words"},
    "medium": {"tokens": 600, "sentences": "6-10 sentences", "words": "200-400 words"}, 
    "long": {"tokens": 1200, "sentences": "10-15 sentences", "words": "400-800 words"}
}

# OpenAI 설정
model="gpt-4"
temperature=0.7
presence_penalty=0.6
frequency_penalty=0.3
```

## 📊 데이터 흐름

### 1. 파일 업로드 프로세스

```
파일 업로드 → TextProcessor → 청킹 → 임베딩 → ChromaDB 저장 → 어휘 추출
```

### 2. 스토리 생성 프로세스

```
키워드 입력 → 벡터 검색 → 컨텍스트 수집 → 프롬프트 생성 → OpenAI API → 품질 검사 → 어휘 분석 → 결과 표시
```

### 3. 어휘 제한 프로세스

```
RAG 어휘 + 필수 문법 단어 → 엄격한 프롬프트 → 생성 → 품질 검사 → 재시도 → 최고 품질 반환
```

## 🎯 핵심 기능

### 1. RAG 어휘 제한
- 업로드된 문서의 단어만 사용
- 필수 문법 단어 허용
- 40% 이상 RAG 어휘 사용률 보장

### 2. 품질 보장
- 자동 재시도 시스템
- 최고 품질 스토리 보존
- 완전 실패 방지

### 3. 어휘 분석
- RAG 외 단어 식별
- 카테고리별 분류
- 사용률 통계

### 4. 다중 파일 지원
- TXT, CSV, PDF 지원
- 교육용 교재 형식 최적화
- 자동 텍스트 정제

## 🚀 성능 최적화

### 1. 임베딩 최적화
- SentenceTransformer 캐싱
- TF-IDF 백업 시스템
- 배치 처리

### 2. 메모리 관리
- 청킹을 통한 메모리 절약
- 임시 파일 자동 정리
- 세션 상태 관리

### 3. API 효율성
- 컨텍스트 길이 제한
- 재시도 로직 최적화
- 에러 핸들링

## 🔍 디버깅 및 모니터링

### 로그 시스템
```python
print(f"🎯 스토리 생성 시작:")
print(f"   - 주요 키워드: {primary_keywords}")
print(f"   - RAG 어휘 제한: {use_vocabulary_restriction}")
print(f"   - 사용 가능한 어휘: {len(available_vocabulary)}개")
```

### 품질 메트릭
- RAG 어휘 사용률
- 키워드 사용률
- 스토리 길이
- 생성 방법 (OpenAI/로컬)

## 📈 향후 개선 방향

### 1. 프롬프트 최적화
- 더 정확한 어휘 제한
- 문맥 인식 강화
- 다국어 지원

### 2. 성능 향상
- 벡터 검색 최적화
- 캐싱 시스템
- 병렬 처리

### 3. 사용자 경험
- 실시간 피드백
- 어휘 제안 시스템
- 스토리 품질 평가

---

*이 문서는 RAG Story Generator 프로젝트의 기술적 구조와 프롬프트 엔지니어링에 대한 상세한 가이드입니다.* 