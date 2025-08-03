import re
from collections import Counter
import sys
sys.path.append('.')
from text_processor import TextProcessor

def analyze_pdf_extraction_detailed(pdf_content: str):
    """PDF에서 추출된 텍스트를 상세 분석"""
    
    print("=" * 60)
    print("📄 PDF 텍스트 추출 상세 분석")
    print("=" * 60)
    
    # 1. 원본 텍스트 정보
    print(f"1. 원본 텍스트 정보:")
    print(f"   - 총 문자 수: {len(pdf_content):,}")
    print(f"   - 줄 수: {len(pdf_content.splitlines()):,}")
    paragraphs = [p for p in pdf_content.split('\n\n') if p.strip()]
    print(f"   - 단락 수: {len(paragraphs):,}")
    
    # 2. 모든 단위 추출 (단어, 숙어, 구문)
    print(f"\n2. 텍스트 단위 분석:")
    
    # 모든 텍스트 토큰 (공백으로 분리)
    all_tokens = pdf_content.split()
    print(f"   - 전체 토큰 수: {len(all_tokens):,}")
    
    # 영어 단어만 추출
    english_words = re.findall(r'\b[a-zA-Z]+\b', pdf_content.lower())
    print(f"   - 영어 단어 수 (중복 포함): {len(english_words):,}")
    
    # 고유 영어 단어
    unique_english_words = set(english_words)
    print(f"   - 고유 영어 단어 수: {len(unique_english_words):,}")
    
    # 숫자가 포함된 단위 (예: "word1", "2nd", "COVID-19")
    alphanumeric = re.findall(r'\b[a-zA-Z0-9]+[a-zA-Z]+[a-zA-Z0-9]*\b', pdf_content.lower())
    print(f"   - 숫자 포함 단위: {len(set(alphanumeric)):,}")
    
    # 하이픈으로 연결된 단어 (예: "long-term", "well-being")
    hyphenated = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*\b', pdf_content.lower())
    print(f"   - 하이픈 연결 단어: {len(set(hyphenated)):,}")
    
    # 아포스트로피 단어 (예: "don't", "it's")
    apostrophe_words = re.findall(r"\b[a-zA-Z]+\'[a-zA-Z]+\b", pdf_content.lower())
    print(f"   - 아포스트로피 단어: {len(set(apostrophe_words)):,}")
    
    # 3. 길이별 분포
    print(f"\n3. 단어 길이 분포:")
    length_dist = Counter(len(word) for word in unique_english_words)
    for length in sorted(length_dist.keys())[:10]:  # 처음 10개 길이만 표시
        print(f"   - {length}글자: {length_dist[length]:,}개")
    if len(length_dist) > 10:
        print(f"   - ... (총 {len(length_dist)}개의 서로 다른 길이)")
    
    # 4. 빈도별 상위 단어들
    print(f"\n4. 가장 자주 나오는 단어 (Top 20):")
    word_freq = Counter(english_words)
    for word, count in word_freq.most_common(20):
        print(f"   - {word}: {count}회")
    
    # 5. 특수 패턴 분석
    print(f"\n5. 특수 패턴 분석:")
    
    # 대문자로만 된 단어 (약어 등)
    all_caps = [word for word in unique_english_words if word.isupper() and len(word) > 1]
    print(f"   - 대문자 단어/약어: {len(all_caps)}개")
    if all_caps[:10]:
        print(f"     예시: {', '.join(all_caps[:10])}")
    
    # 매우 긴 단어들 (10글자 이상)
    long_words = [word for word in unique_english_words if len(word) >= 10]
    print(f"   - 10글자+ 긴 단어: {len(long_words)}개")
    if long_words[:10]:
        print(f"     예시: {', '.join(sorted(long_words)[:10])}")
    
    # 6. 청킹 후 분석
    print(f"\n6. 청킹 과정 분석:")
    tp = TextProcessor()
    clean_text = tp.clean_text(pdf_content)
    chunks = tp.split_text_into_chunks(clean_text)
    
    print(f"   - 정리된 텍스트 길이: {len(clean_text):,}")
    print(f"   - 생성된 청크 수: {len(chunks)}")
    print(f"   - 평균 청크 길이: {sum(len(chunk) for chunk in chunks) / len(chunks):.0f} 문자")
    
    # 각 청크별 단어 수
    chunk_word_counts = []
    for i, chunk in enumerate(chunks):
        words_in_chunk = len(re.findall(r'\b[a-zA-Z]+\b', chunk.lower()))
        chunk_word_counts.append(words_in_chunk)
        if i < 3:  # 처음 3개 청크만 표시
            print(f"   - 청크 {i+1}: {words_in_chunk}개 단어, {len(chunk)}문자")
    
    total_words_in_chunks = sum(chunk_word_counts)
    print(f"   - 청크 내 총 단어 수: {total_words_in_chunks:,}")
    
    # 7. 손실 분석
    print(f"\n7. 데이터 손실 분석:")
    original_word_count = len(english_words)
    chunked_word_count = total_words_in_chunks
    
    print(f"   - 원본 단어 수: {original_word_count:,}")
    print(f"   - 청킹 후 단어 수: {chunked_word_count:,}")
    print(f"   - 손실률: {((original_word_count - chunked_word_count) / original_word_count * 100):.1f}%")
    
    return {
        'total_characters': len(pdf_content),
        'total_tokens': len(all_tokens),
        'english_words': len(english_words),
        'unique_words': len(unique_english_words),
        'chunks': len(chunks),
        'words_in_chunks': total_words_in_chunks
    }

def test_with_sample_text():
    """샘플 텍스트로 테스트"""
    sample_text = """
    This is a sample PDF content with various types of words.
    It contains regular words, hyphenated-words, contractions like don't and won't.
    There are also ACRONYMS, numbers like 123, and mixed cases like iPhone.
    Some phrases might be considered idioms or phrasal verbs like "look up" or "break down".
    Long words like "internationalization" and "antidisestablishmentarianism" are also included.
    """
    
    print("샘플 텍스트 테스트:")
    analyze_pdf_extraction_detailed(sample_text)

if __name__ == "__main__":
    test_with_sample_text()
    
    print("\n" + "="*60)
    print("실제 PDF 텍스트 분석을 위해서는:")
    print("웹앱에서 PDF 업로드 시 이 분석이 자동으로 실행됩니다.")
    print("="*60) 