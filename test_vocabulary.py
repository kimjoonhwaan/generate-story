import re
from collections import Counter

def analyze_vocabulary_extraction(text):
    """텍스트에서 어휘 추출 과정을 분석"""
    
    # 1. 모든 단어 추출
    all_words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    print(f"1. 전체 단어 수 (중복 포함): {len(all_words)}")
    
    # 2. 3글자 이상 단어만 필터링
    words_3plus = [word for word in all_words if len(word) >= 3]
    print(f"2. 3글자 이상 단어 수: {len(words_3plus)}")
    
    # 3. 고유 단어 수
    unique_words = set(words_3plus)
    print(f"3. 고유 단어 수 (중복 제거): {len(unique_words)}")
    
    # 4. 불용어 제거
    stop_words = {'the', 'and', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 
                 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this',
                 'that', 'these', 'those', 'with', 'for', 'from', 'they', 'them',
                 'their', 'there', 'where', 'when', 'what', 'who', 'how', 'why',
                 'but', 'not', 'all', 'any', 'her', 'him', 'his', 'she', 'you',
                 'your', 'our', 'out', 'one', 'two', 'now', 'new', 'old', 'get'}
    
    filtered_words = unique_words - stop_words
    print(f"4. 불용어 제거 후 고유 단어 수: {len(filtered_words)}")
    
    # 5. 가장 자주 나오는 단어들
    word_counts = Counter(words_3plus)
    print(f"\n5. 가장 자주 나오는 단어 (Top 10):")
    for word, count in word_counts.most_common(10):
        status = "❌ 불용어" if word in stop_words else "✅ 유효"
        print(f"   {word}: {count}회 {status}")
    
    # 6. 새로운 유효 단어 샘플
    print(f"\n6. 유효한 단어 샘플 (처음 20개):")
    sample_words = sorted(list(filtered_words))[:20]
    for word in sample_words:
        print(f"   {word}")
    
    return len(all_words), len(words_3plus), len(unique_words), len(filtered_words)

if __name__ == "__main__":
    # 샘플 텍스트로 테스트
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text 
    with many different words. Some words are repeated, and some are very 
    common words that should be filtered out as stop words.
    """
    
    print("=== 어휘 추출 분석 ===")
    analyze_vocabulary_extraction(sample_text)
    
    print("\n" + "="*50)
    print("실제 업로드한 파일의 텍스트를 여기에 붙여넣으면 상세 분석이 가능합니다.") 