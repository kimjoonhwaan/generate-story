import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI
import os
import re
from typing import List, Dict
import json

# Try to import sentence_transformers, fallback to a simpler embedding method
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence_transformers not available, using basic embedding")

class VectorDB:
    def __init__(self, collection_name: str = "rag_documents", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        벡터 DB 초기화
        
        Args:
            collection_name: ChromaDB 컬렉션 이름
            model_name: 임베딩 모델 이름
        """
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.use_sentence_transformers = True
            except Exception as e:
                print(f"Failed to load sentence transformer model: {e}")
                self.use_sentence_transformers = False
                self.model = None
        else:
            self.use_sentence_transformers = False
            self.model = None
            
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # RAG에 등록된 단어들을 저장할 사전
        self.vocabulary = set()
        self._load_vocabulary()
        
        self.azure_embed_client = None
        self.azure_embed_deployment = None

        ak = os.getenv("AOAI_API_KEY")
        ae = os.getenv("AOAI_ENDPOINT")
        av = os.getenv("AOAI_API_VERSION")
        ad = os.getenv("AOAI_EMBEDDING_DEPLOYMENT")
        if ak and ae and av and ad:
            try:
                self.azure_embed_client = AzureOpenAI(
                    api_key=ak,
                    api_version=av,
                    azure_endpoint=ae,
                )
                self.azure_embed_deployment = ad
            except Exception as e:
                print(f"Azure Embedding client init failed: {e}")
                self.azure_embed_client = None
                self.azure_embed_deployment = None
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """
        문서를 벡터 DB에 추가
        
        Args:
            texts: 추가할 텍스트 리스트
            metadatas: 각 텍스트에 대한 메타데이터 리스트
        """
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        # 텍스트를 임베딩으로 변환
        if self.use_sentence_transformers:
            embeddings = self.model.encode(texts).tolist()
        elif self.azure_embed_client:
            embeddings = self._azure_embed(texts)
        else:
            embeddings = self._simple_embedding(texts)
        
        # 고유 ID 생성
        ids = [f"doc_{i}_{hash(text)}" for i, text in enumerate(texts)]
        
        # ChromaDB에 추가
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # 새로운 단어들을 어휘에 추가
        self._extract_and_add_vocabulary(texts)
        
        print(f"{len(texts)}개의 문서가 벡터 DB에 추가되었습니다.")
        print(f"현재 어휘 크기: {len(self.vocabulary)}개 단어")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        쿼리와 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트
        """
        # 쿼리를 임베딩으로 변환
        if self.use_sentence_transformers:
            query_embedding = self.model.encode([query]).tolist()
        elif self.azure_embed_client:
            query_embedding = self._azure_embed([query])
        else:
            query_embedding = self._simple_embedding([query])
        
        # 유사한 문서 검색
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # 결과 포맷팅
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def get_collection_info(self):
        """컬렉션 정보 반환"""
        return {
            'count': self.collection.count(),
            'name': self.collection.name
        }
    
    def clear_collection(self):
        """컬렉션의 모든 데이터 삭제"""
        try:
            # 기존 컬렉션 삭제
            try:
                self.client.delete_collection(self.collection.name)
            except Exception as e:
                print(f"컬렉션 삭제 중 오류 (무시됨): {e}")
            
            # 새 컬렉션 생성
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # 어휘 초기화
            self.vocabulary = set()
            self._save_vocabulary()
            
            print("컬렉션이 초기화되었습니다.")
            
        except Exception as e:
            print(f"컬렉션 초기화 중 오류: {e}")
            # 완전히 새로운 컬렉션 생성 시도
            try:
                import time
                new_collection_name = f"{self.collection.name}_{int(time.time())}"
                self.collection = self.client.get_or_create_collection(
                    name=new_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                self.vocabulary = set()
                self._save_vocabulary()
                print(f"새 컬렉션 '{new_collection_name}'이 생성되었습니다.")
            except Exception as e2:
                print(f"새 컬렉션 생성 실패: {e2}")
                raise e2
    
    def _simple_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        간단한 TF-IDF 기반 임베딩 (sentence_transformers 대안)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        import re
        from collections import Counter
        import math
        
        # 간단한 토큰화
        def tokenize(text):
            return re.findall(r'\b\w+\b', text.lower())
        
        # 모든 텍스트에서 단어 추출
        all_words = set()
        tokenized_texts = []
        for text in texts:
            tokens = tokenize(text)
            tokenized_texts.append(tokens)
            all_words.update(tokens)
        
        all_words = list(all_words)
        vocab_size = len(all_words)
        
        # 각 텍스트에 대해 TF-IDF 벡터 생성
        embeddings = []
        for tokens in tokenized_texts:
            tf = Counter(tokens)
            tf_vector = [tf.get(word, 0) / len(tokens) if tokens else 0 for word in all_words]
            
            # 간단한 정규화 (100차원으로 축소)
            if vocab_size > 100:
                # 상위 100개 차원만 사용
                tf_vector = tf_vector[:100]
            else:
                # 100차원으로 패딩
                tf_vector.extend([0.0] * (100 - len(tf_vector)))
            
            embeddings.append(tf_vector)
        
        return embeddings
    
    def _load_vocabulary(self):
        """저장된 어휘 파일을 로드"""
        vocab_file = "./chroma_db/vocabulary.txt"
        try:
            if os.path.exists(vocab_file):
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    self.vocabulary = set(line.strip().lower() for line in f if line.strip())
                print(f"어휘 로드 완료: {len(self.vocabulary)}개 단어")
        except Exception as e:
            print(f"어휘 로드 오류: {e}")
            self.vocabulary = set()
    
    def _save_vocabulary(self):
        """어휘를 파일에 저장"""
        vocab_file = "./chroma_db/vocabulary.txt"
        try:
            os.makedirs("./chroma_db", exist_ok=True)
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for word in sorted(self.vocabulary):
                    f.write(f"{word}\n")
            print(f"어휘 저장 완료: {len(self.vocabulary)}개 단어")
        except Exception as e:
            print(f"어휘 저장 오류: {e}")
    
    def _extract_and_add_vocabulary(self, texts: List[str]):
        """텍스트에서 단어를 추출하여 어휘에 추가 (교육용 교재 최적화)"""
        import re
        
        initial_vocab_size = len(self.vocabulary)
        new_words = []
        total_extracted_words = 0
        
        for text in texts:
            # 1. 순수 영어 단어
            english_words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            
            # 2. 하이픈 연결 단어 (phrasal verbs, compound words)
            hyphenated_words = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*\b', text.lower())
            
            # 3. 아포스트로피 단어 (contractions)
            apostrophe_words = re.findall(r"\b[a-zA-Z]+\'[a-zA-Z]+\b", text.lower())
            
            # 4. 교육용 교재 형식 처리: "171 attention 주의" → "attention"
            # 숫자로 시작하는 단어에서 앞의 숫자 제거
            numbered_words = re.findall(r'\b\d+[a-zA-Z]+\b', text.lower())
            cleaned_numbered_words = []
            for word in numbered_words:
                # 앞의 숫자 제거하고 영어 부분만 추출
                clean_word = re.sub(r'^\d+', '', word)
                if len(clean_word) >= 2:  # 최소 2글자 이상인 단어만
                    cleaned_numbered_words.append(clean_word)
            
            # 5. 대문자 약어 (예: "USA", "PDF", "API")
            acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
            
            # 모든 단어 합치기
            all_words = english_words + hyphenated_words + apostrophe_words + cleaned_numbered_words + [a.lower() for a in acronyms]
            
            # 중복 제거
            unique_words = list(set(all_words))
            total_extracted_words += len(unique_words)
            
            # 모든 단어를 어휘에 추가
            for word in unique_words:
                if word not in self.vocabulary:
                    new_words.append(word)
                self.vocabulary.add(word)
        
        # 상세한 로그 출력
        print(f"📚 교육용 교재 어휘 추출 결과:")
        print(f"   - 총 추출된 고유 단어 수: {total_extracted_words:,}")
        print(f"   - 새로 추가된 고유 단어 수: {len(new_words):,}")
        print(f"   - 기존 어휘 크기: {initial_vocab_size:,}")
        print(f"   - 현재 어휘 크기: {len(self.vocabulary):,}")
        
        if len(new_words) > 0:
            # 순수 영어 단어들을 우선적으로 표시
            pure_english = [w for w in new_words if re.match(r'^[a-zA-Z]+$', w)]
            print(f"   - 순수 영어 단어 예시 (처음 20개): {sorted(pure_english)[:20]}")
            
            # 카테고리별 분류
            categories = {
                '1글자': [w for w in new_words if len(w) == 1],
                '2글자': [w for w in new_words if len(w) == 2],
                '3-5글자': [w for w in new_words if 3 <= len(w) <= 5],
                '6-10글자': [w for w in new_words if 6 <= len(w) <= 10],
                '10글자+': [w for w in new_words if len(w) > 10],
                '하이픈단어': [w for w in new_words if '-' in w],
                '축약형': [w for w in new_words if "'" in w],
                '대문자약어': [w for w in new_words if w.isupper() and len(w) > 1]
            }
            
            for category, words in categories.items():
                if words:
                    print(f"   - {category}: {len(words)}개 (예: {', '.join(words[:3])})")
        
        # 변경사항 저장
        self._save_vocabulary()
    
    def get_vocabulary(self) -> List[str]:
        """전체 어휘 반환"""
        return sorted(list(self.vocabulary))
    
    def get_filtered_vocabulary(self, keywords: str, context_documents: List[str] = None) -> List[str]:
        """키워드와 관련된 어휘만 필터링하여 반환"""
        # 키워드에서 단어 추출
        keyword_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', keywords.lower()))
        
        # 컨텍스트 문서에서 단어 추출 (있는 경우)
        context_words = set()
        if context_documents:
            for doc in context_documents:
                context_words.update(re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower()))
        
        # 관련 단어들 필터링
        relevant_words = set()
        
        # 키워드 자체 포함
        relevant_words.update(keyword_words & self.vocabulary)
        
        # 컨텍스트에서 추출된 단어들 포함
        relevant_words.update(context_words & self.vocabulary)
        
        # 키워드와 유사한 단어들 추가 (간단한 부분 문자열 매칭)
        for vocab_word in self.vocabulary:
            for keyword in keyword_words:
                if keyword in vocab_word or vocab_word in keyword:
                    relevant_words.add(vocab_word)
                    break
        
        return sorted(list(relevant_words)) 

    # 추가: Azure 임베딩 함수
    def _azure_embed(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = self.azure_embed_client.embeddings.create(
                model=self.azure_embed_deployment,
                input=texts
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            print(f"Azure embedding error, fallback to simple embedding: {e}")
            return self._simple_embedding(texts) 