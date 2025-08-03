import os
import re
from typing import List, Dict
import pandas as pd

# PDF 처리를 위한 라이브러리들
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PDF libraries not available. Install PyPDF2 and pdfplumber for PDF support.")

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        텍스트 처리기 초기화
        
        Args:
            chunk_size: 각 청크의 최대 문자 수
            chunk_overlap: 청크 간 겹치는 문자 수
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def read_file(self, file_path: str) -> str:
        """
        파일에서 텍스트 읽기 (PDF 지원 포함)
        
        Args:
            file_path: 읽을 파일 경로
            
        Returns:
            파일 내용
        """
        try:
            # 파일 확장자에 따른 처리
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
                return df.to_string(index=False)
            elif file_ext == '.pdf':
                return self._read_pdf(file_path)
            else:
                # 기본적으로 텍스트 파일로 처리
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
                    
        except Exception as e:
            print(f"파일 읽기 오류: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        텍스트 정리
        
        Args:
            text: 정리할 텍스트
            
        Returns:
            정리된 텍스트
        """
        # 여러 줄바꿈을 하나로 변경
        text = re.sub(r'\n+', '\n', text)
        
        # 여러 공백을 하나로 변경
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            분할된 텍스트 청크 리스트
        """
        # 문장 단위로 먼저 분할
        sentences = re.split(r'[.!?]\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 현재 청크에 문장을 추가했을 때의 길이 확인
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # 현재 청크를 저장하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 겹치는 부분이 있는 청크 생성 (더 나은 검색을 위해)
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            overlapped_chunks.append(chunk)
            
            # 다음 청크와 겹치는 부분 생성
            if i < len(chunks) - 1 and self.chunk_overlap > 0:
                overlap_text = chunk[-self.chunk_overlap:] + " " + chunks[i+1][:self.chunk_overlap]
                overlapped_chunks.append(overlap_text.strip())
        
        return [chunk for chunk in overlapped_chunks if chunk.strip()]
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract English keywords from text
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        # Keep only English letters, numbers, and spaces
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Split into words
        words = clean_text.split()
        
        # Remove short words (less than 3 characters for English)
        keywords = [word for word in words if len(word) >= 3]
        
        # Common English stop words to exclude
        stop_words = {'the', 'and', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 
                     'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this',
                     'that', 'these', 'those', 'with', 'for', 'from', 'they', 'them',
                     'their', 'there', 'where', 'when', 'what', 'who', 'how', 'why',
                     'but', 'not', 'all', 'any', 'her', 'him', 'his', 'she', 'you',
                     'your', 'our', 'out', 'one', 'two', 'now', 'new', 'old', 'get'}
        
        # Remove stop words, duplicates, and convert to lowercase
        keywords = list(set([word.lower() for word in keywords if word.lower() not in stop_words]))
        
        return keywords
    
    def process_file(self, file_path: str) -> Dict:
        """
        파일을 처리하여 청크와 메타데이터 생성
        
        Args:
            file_path: 처리할 파일 경로
            
        Returns:
            처리 결과 딕셔너리
        """
        # 파일 읽기
        text = self.read_file(file_path)
        
        if not text:
            return {"chunks": [], "metadata": []}
        
        # 텍스트 정리
        clean_text = self.clean_text(text)
        
        # 청크로 분할
        chunks = self.split_text_into_chunks(clean_text)
        
        # 각 청크에 대한 메타데이터 생성
        metadata = []
        for i, chunk in enumerate(chunks):
            keywords = self.extract_keywords(chunk)
            # ChromaDB는 메타데이터 값으로 리스트를 허용하지 않으므로 문자열로 변환
            keywords_str = ", ".join(keywords) if keywords else ""
            metadata.append({
                "source": file_path,
                "chunk_id": i,
                "keywords": keywords_str,
                "chunk_length": len(chunk)
            })
        
        return {
            "chunks": chunks,
            "metadata": metadata
        }
    
    def _read_pdf(self, file_path: str) -> str:
        """
        PDF 파일에서 텍스트 추출
        
        Args:
            file_path: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        if not PDF_AVAILABLE:
            print("PDF 라이브러리가 설치되지 않았습니다. pip install PyPDF2 pdfplumber")
            return ""
        
        text = ""
        
        # 방법 1: pdfplumber 사용 (더 정확한 텍스트 추출)
        try:
            with pdfplumber.open(file_path) as pdf:
                print(f"📄 PDF 분석 시작: {len(pdf.pages)}페이지")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        print(f"   페이지 {page_num}: {len(page_text)} 문자")
            
            if text.strip():
                print(f"✅ pdfplumber로 PDF 읽기 성공: {len(text):,} 문자")
                
                # 상세 분석 실행
                self._analyze_pdf_content(text)
                return text
                
        except Exception as e:
            print(f"pdfplumber 오류: {e}")
        
        # 방법 2: PyPDF2 사용 (백업)
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                print(f"✅ PyPDF2로 PDF 읽기 성공: {len(text):,} 문자")
                self._analyze_pdf_content(text)
                return text
                
        except Exception as e:
            print(f"PyPDF2 오류: {e}")
        
        print("❌ PDF 텍스트 추출 실패")
        return ""
    
    def _analyze_pdf_content(self, pdf_content: str):
        """PDF 내용 상세 분석"""
        try:
            print(f"\n📊 PDF 내용 상세 분석:")
            
            # 샘플 내용 표시 (처음 500자)
            sample_content = pdf_content[:500].replace('\n', ' ').strip()
            print(f"   - 샘플 내용 (처음 500자):")
            print(f"     '{sample_content}...'")
            print()
            
            # 모든 텍스트 토큰 (공백으로 분리)
            all_tokens = pdf_content.split()
            print(f"   - 전체 토큰 수: {len(all_tokens):,}")
            
            # 영어 단어만 추출
            english_words = re.findall(r'\b[a-zA-Z]+\b', pdf_content.lower())
            print(f"   - 영어 단어 수 (중복 포함): {len(english_words):,}")
            
            # 고유 영어 단어
            unique_english_words = set(english_words)
            print(f"   - 고유 영어 단어 수: {len(unique_english_words):,}")
            
            # 하이픈으로 연결된 단어 (phrasal verbs, compound words)
            hyphenated = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*\b', pdf_content.lower())
            hyphenated_unique = set(hyphenated)
            print(f"   - 하이픈 연결 단어: {len(hyphenated_unique):,}")
            if hyphenated_unique:
                print(f"     예시: {', '.join(list(hyphenated_unique)[:5])}")
            
            # 아포스트로피 단어 (contractions)
            apostrophe_words = re.findall(r"\b[a-zA-Z]+\'[a-zA-Z]+\b", pdf_content.lower())
            apostrophe_unique = set(apostrophe_words)
            print(f"   - 축약형 단어: {len(apostrophe_unique):,}")
            if apostrophe_unique:
                print(f"     예시: {', '.join(list(apostrophe_unique)[:5])}")
            
            # 숫자가 포함된 단위
            alphanumeric = re.findall(r'\b[a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*\b', pdf_content.lower())
            alphanumeric_unique = set(alphanumeric) - unique_english_words  # 순수 영어 단어 제외
            print(f"   - 숫자 포함 단위: {len(alphanumeric_unique):,}")
            if alphanumeric_unique:
                print(f"     예시: {', '.join(list(alphanumeric_unique)[:10])}")
            
            # 대문자 단어/약어
            all_caps = re.findall(r'\b[A-Z]{2,}\b', pdf_content)
            print(f"   - 대문자 단어/약어: {len(set(all_caps)):,}")
            if all_caps:
                print(f"     예시: {', '.join(list(set(all_caps))[:5])}")
            
            # 길이별 분포
            from collections import Counter
            length_dist = Counter(len(word) for word in unique_english_words)
            print(f"   - 단어 길이 분포: 1글자({length_dist[1]}), 2글자({length_dist[2]}), 3글자({length_dist[3]}), 4글자({length_dist[4]}), 5글자({length_dist[5]}), 6+글자({sum(count for length, count in length_dist.items() if length >= 6)})")
            
            # 청킹 후 분석
            clean_text = self.clean_text(pdf_content)
            chunks = self.split_text_into_chunks(clean_text)
            
            chunk_word_counts = []
            for chunk in chunks:
                words_in_chunk = len(re.findall(r'\b[a-zA-Z]+\b', chunk.lower()))
                chunk_word_counts.append(words_in_chunk)
            
            total_words_in_chunks = sum(chunk_word_counts)
            
            print(f"   - 생성된 청크 수: {len(chunks)}")
            print(f"   - 청크 내 총 단어 수: {total_words_in_chunks:,}")
            
            # 손실 분석
            original_word_count = len(english_words)
            loss_rate = ((original_word_count - total_words_in_chunks) / original_word_count * 100) if original_word_count > 0 else 0
            print(f"   - 처리 과정 손실률: {loss_rate:.1f}%")
            
            # 가능한 총 어휘 수 계산
            total_vocabulary = len(unique_english_words) + len(hyphenated_unique) + len(apostrophe_unique) + len(alphanumeric_unique) + len(set(all_caps))
            print(f"   - 예상 총 어휘 수: {total_vocabulary:,} (영어단어 + 하이픈단어 + 축약형 + 숫자포함 + 약어)")
            
            print()
            
        except Exception as e:
            print(f"PDF 분석 중 오류: {e}") 