# 🚀 RAG Story Generator 배포 가이드

이 문서는 RAG Story Generator 앱을 다양한 플랫폼에 배포하는 방법을 설명합니다.

## 📋 **배포 옵션**

### 1. **Streamlit Cloud (추천 - 가장 쉬움)**

#### 장점:
- ✅ 무료 플랜 제공
- ✅ GitHub 연동으로 자동 배포
- ✅ 쉬운 설정
- ✅ SSL 인증서 자동 제공

#### 단점:
- ❌ 세션당 제한 (무료 플랜)
- ❌ 저장소 크기 제한
- ❌ API 키 노출 위험

#### 배포 단계:

1. **GitHub 저장소 준비**
   ```bash
   # 이미 완료됨
   git push origin master
   ```

2. **Streamlit Cloud 설정**
   - [share.streamlit.io](https://share.streamlit.io) 방문
   - GitHub 계정으로 로그인
   - "New app" 클릭
   - 저장소 선택: `kimjoonhwaan/generate-story`
   - Main file path: `streamlit_app.py`
   - Deploy 클릭

3. **환경변수 설정**
   - App settings → Secrets
   - 다음 내용 추가:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

### 2. **Docker + 클라우드 플랫폼**

#### 장점:
- ✅ 완전한 제어권
- ✅ 확장성
- ✅ 보안성

#### 단점:
- ❌ 복잡한 설정
- ❌ 비용 발생

#### 배포 단계:

1. **로컬에서 Docker 빌드**
   ```bash
   docker build -t rag-story-generator .
   docker run -p 8501:8501 -e OPENAI_API_KEY=your_key rag-story-generator
   ```

2. **Docker Compose 사용**
   ```bash
   # .env 파일 생성
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   
   # 실행
   docker-compose up -d
   ```

3. **클라우드 플랫폼 배포**
   - **AWS**: ECS, EKS
   - **Google Cloud**: Cloud Run, GKE
   - **Azure**: Container Instances, AKS
   - **DigitalOcean**: App Platform, Droplets

### 3. **Heroku**

#### 장점:
- ✅ 쉬운 배포
- ✅ 무료 플랜 (제한적)
- ✅ Git 연동

#### 단점:
- ❌ 메모리 제한
- ❌ 슬립 모드 (무료 플랜)

#### 배포 단계:

1. **Heroku CLI 설치**
   ```bash
   # Windows
   winget install --id=Heroku.HerokuCLI
   ```

2. **Heroku 앱 생성**
   ```bash
   heroku login
   heroku create your-rag-app-name
   ```

3. **환경변수 설정**
   ```bash
   heroku config:set OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **배포**
   ```bash
   git push heroku master
   ```

### 4. **Railway**

#### 장점:
- ✅ 간단한 배포
- ✅ GitHub 연동
- ✅ 합리적인 가격

#### 배포 단계:

1. [railway.app](https://railway.app) 방문
2. GitHub 계정으로 로그인
3. "New Project" → "Deploy from GitHub repo"
4. 저장소 선택
5. 환경변수 설정

### 5. **Vercel (제한적)**

#### 장점:
- ✅ 빠른 배포
- ✅ 무료 플랜

#### 단점:
- ❌ Streamlit 지원 제한적
- ❌ 서버리스 환경

## 🔧 **배포 전 준비사항**

### 1. **환경변수 설정**
```bash
# .env 파일 (로컬 개발용)
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. **의존성 확인**
```bash
pip install -r requirements.txt
```

### 3. **테스트**
```bash
python test_rag.py
streamlit run streamlit_app.py
```

## 🛡️ **보안 고려사항**

### 1. **API 키 보호**
- ✅ 환경변수 사용
- ✅ .env 파일을 .gitignore에 추가
- ❌ 코드에 하드코딩 금지

### 2. **파일 업로드 제한**
- ✅ 파일 크기 제한
- ✅ 파일 타입 검증
- ✅ 악성 파일 스캔

### 3. **데이터베이스 보안**
- ✅ ChromaDB 파일 권한 설정
- ✅ 백업 정책 수립

## 📊 **성능 최적화**

### 1. **메모리 사용량**
- 벡터 DB 크기 모니터링
- 불필요한 세션 데이터 정리

### 2. **응답 시간**
- 임베딩 모델 캐싱
- 쿼리 결과 캐싱

### 3. **확장성**
- 로드 밸런싱
- 데이터베이스 분산

## 🔍 **모니터링**

### 1. **로그 관리**
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### 2. **성능 메트릭**
- 응답 시간
- 메모리 사용량
- 에러율

### 3. **사용자 분석**
- 페이지 뷰
- 기능 사용률
- 사용자 피드백

## 🚨 **문제 해결**

### 1. **일반적인 오류**
- **ImportError**: 의존성 설치 확인
- **MemoryError**: 메모리 제한 증가
- **TimeoutError**: 요청 시간 제한 조정

### 2. **디버깅**
```bash
# 로그 확인
docker logs container_name

# 환경변수 확인
echo $OPENAI_API_KEY
```

## 📞 **지원**

문제가 발생하면:
1. GitHub Issues 생성
2. 로그 파일 첨부
3. 환경 정보 제공

---

**추천 배포 순서:**
1. **Streamlit Cloud** (테스트용)
2. **Railway** (프로덕션용)
3. **Docker + 클라우드** (대규모 서비스용) 