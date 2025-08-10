import os
from typing import List, Dict
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
import random

load_dotenv()

class StoryGenerator:
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        self.client = None
        self.is_azure = False
        self.azure_deployment = None

        if not use_openai:
            return

        # 1) Azure OpenAI 우선
        aoai_key = os.getenv("AOAI_API_KEY")
        aoai_endpoint = os.getenv("AOAI_ENDPOINT")
        aoai_version = os.getenv("AOAI_API_VERSION")
        aoai_deploy = os.getenv("AOAI_DEPLOY_GPT4O")

        if aoai_key and aoai_endpoint and aoai_version and aoai_deploy:
            self.client = AzureOpenAI(
                api_key=aoai_key,
                api_version=aoai_version,
                azure_endpoint=aoai_endpoint,
            )
            self.is_azure = True
            self.azure_deployment = aoai_deploy
            return

        # 2) 일반 OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            print("Warning: No OPENAI_API_KEY or AOAI_API_KEY found. Switching to local generation.")
            self.use_openai = False
    
    def generate_story_with_openai(self, keywords: List[str], context_documents: List[str] = None, 
                                   story_length: str = "medium", available_vocabulary: List[str] = None) -> str:
        if not self.client:
            return f"OpenAI API client not initialized. Please set OPENAI_API_KEY environment variable."

        # 길이 설정
        length_settings = {
            "short": {"tokens": 300, "sentences": "3-5 sentences", "words": "100-200 words"},
            "medium": {"tokens": 600, "sentences": "6-10 sentences", "words": "200-400 words"}, 
            "long": {"tokens": 1200, "sentences": "10-15 sentences", "words": "400-800 words"}
        }
        settings = length_settings.get(story_length, length_settings["medium"])
        
        # 컨텍스트 길이 제한
        context_text = ""
        if context_documents:
            limited_docs = []
            for doc in context_documents[:3]:
                limited_doc = doc[:200] + "..." if len(doc) > 200 else doc
                limited_docs.append(limited_doc)
            context_text = "\n".join(limited_docs)
        
        # 키워드 분류
        primary_keywords = keywords[:3]
        secondary_keywords = keywords[3:] if len(keywords) > 3 else []
        
        # RAG 어휘 제한 프롬프트 - 매우 엄격하게
        vocabulary_instruction = ""
        if available_vocabulary:
            # 실제 사용 가능한 어휘를 더 많이 표시
            vocab_sample = available_vocabulary[:100]  # 더 많은 샘플 표시
            vocabulary_instruction = f"""
**CRITICAL VOCABULARY RESTRICTION - YOU MUST FOLLOW THIS EXACTLY:**

**ONLY USE THESE WORDS:**
1. RAG Vocabulary: {', '.join(vocab_sample)}
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
        
        prompt = f"""Create an engaging English story using these requirements:

**PRIMARY KEYWORDS (MUST include all):** {', '.join(primary_keywords)}
**SECONDARY KEYWORDS (include if possible):** {', '.join(secondary_keywords) if secondary_keywords else 'None'}

**STORY LENGTH:** {settings['sentences']} ({settings['words']})

**CONTEXT INFORMATION:**
{context_text if context_text else 'Use creativity to build context around the keywords.'}

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

        # OpenAI API 재시도 로직
        max_retries = 3
        best_story = None
        best_rag_rate = 0
        
        for attempt in range(max_retries):
            try:
                print(f"   - OpenAI API 시도 {attempt + 1}/{max_retries}...")
                
                response = self.client.chat.completions.create(
                   # model="gpt-4",
                    model=(self.azure_deployment if self.is_azure else "gpt-4"),
                    messages=[
                        {"role": "system", "content": "You are a story writer who MUST follow vocabulary restrictions EXACTLY. When given a vocabulary list, you can ONLY use words from that list plus basic grammar words (a, an, the, is, are, was, were, etc.). If you cannot express something with the allowed words, you MUST rephrase or find alternatives from the vocabulary list. This is a strict vocabulary exercise."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=settings["tokens"],
                    temperature=0.7,
                    presence_penalty=0.6,
                    frequency_penalty=0.3
                )
                
                story = response.choices[0].message.content.strip()
                
                # 간단한 품질 검사
                if len(story.split()) < 50:
                    print(f"   - 시도 {attempt + 1}: 너무 짧은 스토리, 재시도...")
                    continue
                
                # 의미없는 반복 확인
                words = story.lower().split()
                if words.count('thing') > len(words) * 0.1 or words.count('something') > len(words) * 0.1:
                    print(f"   - 시도 {attempt + 1}: 품질 불량 (반복 단어 과다), 재시도...")
                    continue
                
                # RAG 어휘 사용률 확인 (어휘 제한이 있을 때)
                if available_vocabulary:
                    vocab_set = set(w.lower() for w in available_vocabulary)
                    essential_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 
                                     'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 
                                     'must', 'shall', 'and', 'or', 'but', 'so', 'if', 'when', 'where', 'what', 
                                     'who', 'how', 'why', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 
                                     'about', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 
                                     'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 
                                     'that', 'these', 'those', 'here', 'there', 'now', 'then'}
                    allowed_words = vocab_set | essential_words
                    
                    rag_words_used = sum(1 for word in words if word in vocab_set)
                    total_content_words = len([w for w in words if w not in essential_words])
                    
                    if total_content_words > 0:
                        rag_usage_rate = rag_words_used / total_content_words
                        
                        # 최고 품질 스토리 저장
                        if rag_usage_rate > best_rag_rate:
                            best_story = story
                            best_rag_rate = rag_usage_rate
                        
                        if rag_usage_rate < 0.4:  # 40% 미만이면 재시도
                            print(f"   - 시도 {attempt + 1}: RAG 어휘 사용률 낮음 ({rag_usage_rate:.1%}), 재시도...")
                            continue
                
                print(f"   ✅ OpenAI API 생성 성공 (시도 {attempt + 1})")
                
                # 어휘 제한이 있으면 RAG에 없는 단어에 주석 추가
                if available_vocabulary:
                    story = self._annotate_non_rag_words(story, available_vocabulary)
                
                return story
                
            except Exception as e:
                print(f"   - OpenAI API 시도 {attempt + 1} 실패: {e}")
                if attempt == max_retries - 1:
                    break
                continue
        
        # 모든 시도 실패 시 최고 품질 스토리 반환
        if best_story:
            print(f"   ⚠️ 최고 품질 스토리 반환 (RAG 사용률: {best_rag_rate:.1%})")
            if available_vocabulary:
                best_story = self._annotate_non_rag_words(best_story, available_vocabulary)
            return best_story
        
        return "Failed to generate story after multiple attempts."
    
    def _annotate_non_rag_words(self, story: str, available_vocabulary: List[str]) -> str:
        """
        RAG 어휘에 없는 단어들을 수집하여 별도로 표시
        """
        if not available_vocabulary:
            return story
        
        import re
        
        # RAG 어휘를 소문자로 변환
        vocab_set = set(w.lower() for w in available_vocabulary)
        
        # 항상 허용되는 필수 문법 단어들
        essential_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 
            'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 
            'must', 'shall', 'and', 'or', 'but', 'so', 'if', 'when', 'where', 'what', 
            'who', 'how', 'why', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 
            'about', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 
            'that', 'these', 'those', 'here', 'there', 'now', 'then', 'not', 'very', 
            'too', 'also', 'only', 'just', 'even', 'still', 'yet', 'already', 'always', 
            'never', 'often', 'sometimes', 'one', 'two', 'three', 'four', 'five'
        }
        
        # 허용된 단어 집합 (RAG 어휘 + 필수 문법 단어)
        allowed_words = vocab_set | essential_words
        
        # 스토리에서 모든 단어 추출
        all_words = re.findall(r'\b[a-zA-Z]+\b', story)
        non_rag_words = []
        
        for word in all_words:
            if word.lower() not in allowed_words:
                non_rag_words.append(word)
        
        # 중복 제거하고 정렬
        unique_non_rag_words = sorted(list(set(non_rag_words)))
        
        # 결과 구성
        result = story
        
        # 통계 정보 추가
        total_words = len(all_words)
        rag_words_used = len([w for w in all_words if w.lower() in vocab_set])
        non_rag_count = len(unique_non_rag_words)
        
        if non_rag_count > 0:
            result += f"\n\n<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #ff6b6b; margin: 20px 0;'>"
            result += f"<h4 style='color: #ff6b6b; margin-top: 0;'>📊 어휘 분석 결과</h4>"
            result += f"<p><strong>전체 단어:</strong> {total_words}개</p>"
            result += f"<p><strong>RAG 어휘 사용:</strong> {rag_words_used}개 ({rag_words_used/total_words:.1%})</p>"
            result += f"<p><strong>RAG에 없는 단어:</strong> {non_rag_count}개 ({non_rag_count/total_words:.1%})</p>"
            
            if non_rag_count > 0:
                result += f"<details style='margin-top: 10px;'>"
                result += f"<summary style='cursor: pointer; color: #ff6b6b; font-weight: bold;'>🔍 RAG에 없는 단어 목록 ({non_rag_count}개)</summary>"
                result += f"<div style='margin-top: 10px; padding: 10px; background-color: #fff; border-radius: 5px; border: 1px solid #ddd;'>"
                
                # 단어들을 카테고리별로 분류
                categories = {
                    '명사': [],
                    '동사': [],
                    '형용사': [],
                    '부사': [],
                    '기타': []
                }
                
                for word in unique_non_rag_words:
                    # 간단한 품사 분류 (정확하지 않을 수 있음)
                    if word.endswith(('ing', 'ed', 's')):
                        categories['동사'].append(word)
                    elif word.endswith(('ly', 'fully')):
                        categories['부사'].append(word)
                    elif word.endswith(('al', 'ful', 'ous', 'ive', 'able', 'ible')):
                        categories['형용사'].append(word)
                    elif len(word) > 4:
                        categories['명사'].append(word)
                    else:
                        categories['기타'].append(word)
                
                for category, words in categories.items():
                    if words:
                        result += f"<p><strong>{category}:</strong> {', '.join(words)}</p>"
                
                result += f"</div></details>"
            
            result += f"</div>"
        
        return result
    
    def generate_story_locally(self, keywords: List[str], context_documents: List[str] = None,
                              story_length: str = "medium", available_vocabulary: List[str] = None) -> str:
        """
        Generate a story using local templates (disabled when vocabulary restriction is enabled)
        """
        # 어휘 제한이 있으면 로컬 생성 비활성화
        if available_vocabulary:
            return "Local generation is disabled when vocabulary restriction is enabled. Please use OpenAI API."
        
        # 길이 설정
        length_settings = {
            "short": {"sentences": 3, "words_per_sentence": 12},
            "medium": {"sentences": 6, "words_per_sentence": 15}, 
            "long": {"sentences": 10, "words_per_sentence": 18}
        }
        settings = length_settings.get(story_length, length_settings["medium"])
        
        # 키워드 처리
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]
        
        primary_keywords = keywords[:3]
        
        # 컨텍스트에서 관련 단어 추출
        context_words = []
        if context_documents:
            for doc in context_documents[:2]:  # 상위 2개 문서만 사용
                context_words.extend(self.extract_relevant_words(doc)[:5])
        
        # 이야기 템플릿 - 더 동적이고 키워드 중심적으로
        story_templates = [
            f"[LOCAL]",
            f"In a world where {primary_keywords[0] if len(primary_keywords) > 0 else 'adventure'} was everything, a young person discovered something extraordinary.",
            f"The story begins with an unexpected encounter involving {primary_keywords[0] if len(primary_keywords) > 0 else 'mystery'}.",
            f"Once upon a time, {primary_keywords[0] if len(primary_keywords) > 0 else 'magic'} changed someone's life forever.",
            f"In the heart of the city, {primary_keywords[0] if len(primary_keywords) > 0 else 'discovery'} led to an amazing journey."
        ]
        
        # 스토리 문장들 구성
        story_sentences = []
        
        # 시작 문장
        story_sentences.append(random.choice(story_templates))
        
        # 중간 문장들
        middle_templates = [
            f"The journey involving {primary_keywords[0] if len(primary_keywords) > 0 else 'exploration'} was filled with challenges and discoveries.",
            f"Every step forward revealed more about {primary_keywords[1] if len(primary_keywords) > 1 else 'the mystery'}.",
            f"Understanding {primary_keywords[2] if len(primary_keywords) > 2 else 'the situation'} required patience and wisdom.",
            f"Each step forward involved {primary_keywords[0] if len(primary_keywords) > 0 else 'careful planning'} and dedication.",
            f"The community gathered to discuss {', '.join(primary_keywords[:2])}.",
            f"Success came through combining {primary_keywords[0] if len(primary_keywords) > 0 else 'skill'} with determination."
        ]
        
        # 필요한 만큼 중간 문장 추가
        for i in range(settings["sentences"] - 2):  # 시작과 끝 문장 제외
            if context_words and random.random() > 0.5:
                context_word = random.choice(context_words)
                sentence = f"The {context_word} played an important role in understanding {random.choice(primary_keywords)}."
            else:
                sentence = random.choice(middle_templates)
            story_sentences.append(sentence)
        
        # 마무리 문장
        ending_templates = [
            f"In the end, {primary_keywords[0] if len(primary_keywords) > 0 else 'the journey'} taught everyone the value of perseverance.",
            f"The story of {', '.join(primary_keywords[:2])} became a legend that inspired many.",
            f"Through {primary_keywords[0] if len(primary_keywords) > 0 else 'this experience'}, a new understanding was born.",
            f"The lesson about {', '.join(primary_keywords)} would be remembered forever.",
            f"And so, the adventure involving {primary_keywords[0] if len(primary_keywords) > 0 else 'discovery'} came to a meaningful conclusion."
        ]
        
        story_sentences.append(random.choice(ending_templates))
        
        return " ".join(story_sentences)
    
    def extract_relevant_words(self, text: str) -> List[str]:
        """
        Extract relevant English words from text for story enhancement
        
        Args:
            text: Input text
            
        Returns:
            List of relevant words
        """
        import re
        
        # Extract words (only alphabetic characters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common words to exclude
        stop_words = {'the', 'and', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 
                     'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this',
                     'that', 'these', 'those', 'with', 'for', 'from', 'they', 'them',
                     'their', 'there', 'where', 'when', 'what', 'who', 'how', 'why'}
        
        # Filter out stop words and get unique words
        relevant_words = list(set([word for word in words if word not in stop_words]))
        
        return relevant_words[:10]  # Return top 10 relevant words
    
    def generate_story(self, keywords: str, context_documents: List[str] = None,
                      story_length: str = "medium", use_vocabulary_restriction: bool = False, 
                      available_vocabulary: List[str] = None) -> Dict:
        """
        Generate a story with detailed logging and metadata
        """
        # 키워드 처리
        if isinstance(keywords, str):
            keyword_list = [k.strip() for k in keywords.split(',')]
        else:
            keyword_list = keywords
        
        # 로깅
        print(f"🎯 스토리 생성 시작:")
        print(f"   - 주요 키워드: {', '.join(keyword_list[:3])}")
        print(f"   - 전체 키워드: {', '.join(keyword_list)}")
        print(f"   - 길이: {story_length}")
        print(f"   - RAG 어휘 제한: {use_vocabulary_restriction}")
        
        if context_documents:
            print(f"   - 관련 컨텍스트 문서: {len(context_documents)}/3")
        
        if use_vocabulary_restriction and available_vocabulary:
            print(f"   - 사용 가능한 어휘: {len(available_vocabulary)}개")
            
            # 어휘 제한이 있으면 OpenAI만 사용
            story = self.generate_story_with_openai(
                keyword_list, context_documents, story_length, available_vocabulary
            )
            method = "openai"
        else:
            # 일반 모드: OpenAI 우선, 실패 시 로컬
            try:
                story = self.generate_story_with_openai(
                    keyword_list, context_documents, story_length, None
                )
                method = "openai"
            except Exception as e:
                print(f"   - OpenAI 실패, 로컬 생성으로 전환: {e}")
                story = self.generate_story_locally(
                    keyword_list, context_documents, story_length, None
                )
                method = "local"
        
        # 결과 분석
        word_count = len(story.split())
        used_keywords = []
        story_lower = story.lower()
        
        for keyword in keyword_list:
            if keyword.lower() in story_lower:
                used_keywords.append(keyword)
        
        keyword_usage_rate = len(used_keywords) / len(keyword_list) if keyword_list else 0
        
        print(f"📊 생성 결과:")
        print(f"   - 방법: {method}")
        print(f"   - 단어 수: {word_count}")
        print(f"   - 사용된 키워드: {', '.join(used_keywords)} ({len(used_keywords)}/{len(keyword_list)})")
        
        return {
            'story': story,
            'method': method,
            'word_count': word_count,
            'keywords_used': used_keywords,
            'keyword_usage_rate': keyword_usage_rate,
            'vocabulary_restricted': use_vocabulary_restriction,
            'vocabulary_count': len(available_vocabulary) if available_vocabulary else 0,
            'context_documents_count': len(context_documents) if context_documents else 0
        }
    
    def enhance_story_with_context(self, base_story: str, context_documents: List[str]) -> str:
        """
        Enhance an existing story with additional context
        
        Args:
            base_story: Base story to enhance
            context_documents: Additional context documents
            
        Returns:
            Enhanced story
        """
        if not context_documents:
            return base_story
        
        # Extract themes from context
        context_words = []
        for doc in context_documents:
            context_words.extend(self.extract_relevant_words(doc))
        
        # Add context-based enhancement
        if context_words:
            enhancement = f"\n\nThe story was inspired by themes of {', '.join(context_words[:5])}, creating a rich tapestry of narrative elements."
            return base_story + enhancement
        
        return base_story
    
    def _generate_constrained_story(self, keywords: List[str], available_vocabulary: List[str], 
                                   context_documents: List[str] = None, settings: dict = None) -> str:
        """
        제약된 어휘로 자연스러운 스토리 생성 (대폭 개선)
        
        Args:
            keywords: 키워드 리스트
            available_vocabulary: 사용 가능한 어휘
            context_documents: 컨텍스트 문서
            settings: 길이 설정
            
        Returns:
            생성된 스토리
        """
        if not available_vocabulary:
            return "No vocabulary available for story generation."
        
        # 기본 설정
        if not settings:
            settings = {"sentences": 8, "words_per_sentence": 18}
        
        # 필수 문법 단어들
        essential_words = {
            'articles': ['a', 'an', 'the'],
            'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 
                        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'],
            'prepositions': ['in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 'about', 'over', 'under', 
                           'through', 'during', 'before', 'after', 'between', 'among', 'into', 'onto'],
            'conjunctions': ['and', 'but', 'or', 'so', 'because', 'when', 'while', 'if', 'then', 'than', 'as'],
            'be_verbs': ['is', 'am', 'are', 'was', 'were', 'be', 'been', 'being'],
            'auxiliary': ['have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 
                         'should', 'may', 'might', 'must', 'shall'],
            'basic_verbs': ['get', 'got', 'go', 'went', 'come', 'came', 'make', 'made', 'take', 'took'],
            'adverbs': ['not', 'very', 'also', 'just', 'even', 'still', 'always', 'never', 'often', 'sometimes'],
            'question': ['what', 'who', 'where', 'when', 'why', 'how'],
            'numbers': ['one', 'two', 'three', 'first', 'second', 'last', 'next', 'many', 'few', 'some', 'all']
        }
        
        # 모든 필수 단어를 하나의 리스트로 합치기
        all_essential = []
        for category in essential_words.values():
            all_essential.extend(category)
        
        # 사용 가능한 어휘를 카테고리별로 분류
        vocab_categories = {
            'characters': [],
            'actions': [],
            'objects': [],
            'places': [],
            'descriptors': [],
            'emotions': [],
            'time': []
        }
        
        # 어휘 분류
        for word in available_vocabulary:
            word_lower = word.lower()
            if word_lower in ['student', 'teacher', 'person', 'people', 'friend', 'family', 'child', 'adult', 'man', 'woman']:
                vocab_categories['characters'].append(word_lower)
            elif word_lower in ['teach', 'learn', 'study', 'read', 'write', 'work', 'play', 'run', 'walk', 'help', 'start', 'finish', 'think', 'know']:
                vocab_categories['actions'].append(word_lower)
            elif word_lower in ['school', 'library', 'classroom', 'home', 'park', 'street', 'building', 'place']:
                vocab_categories['places'].append(word_lower)
            elif word_lower in ['book', 'lesson', 'computer', 'phone', 'car', 'table', 'chair', 'paper']:
                vocab_categories['objects'].append(word_lower)
            elif word_lower in ['happy', 'sad', 'excited', 'nervous', 'proud', 'angry', 'calm']:
                vocab_categories['emotions'].append(word_lower)
            elif word_lower in ['good', 'bad', 'big', 'small', 'new', 'old', 'important', 'difficult', 'easy']:
                vocab_categories['descriptors'].append(word_lower)
            elif word_lower in ['today', 'yesterday', 'tomorrow', 'morning', 'afternoon', 'evening', 'day', 'week', 'year']:
                vocab_categories['time'].append(word_lower)
            else:
                # 기본적으로 객체나 설명어로 분류
                if len(word_lower) > 6:
                    vocab_categories['descriptors'].append(word_lower)
                else:
                    vocab_categories['objects'].append(word_lower)
        
        # 키워드를 주요 요소로 활용
        primary_keywords = keywords[:3]
        
        # 스토리 문장들 생성
        story_sentences = []
        
        # 캐릭터 선택
        character = vocab_categories['characters'][0] if vocab_categories['characters'] else 'person'
        place = vocab_categories['places'][0] if vocab_categories['places'] else primary_keywords[0] if primary_keywords else 'place'
        
        # 스토리 구조화된 생성
        sentences_per_part = {
            'opening': 2,
            'development': settings['sentences'] - 4,
            'climax': 1,
            'conclusion': 1
        }
        
        # 1. 시작 부분
        opening_templates = [
            f"The {character} was in the {place}.",
            f"Every day, the {character} would think about {primary_keywords[0] if primary_keywords else 'life'}."
        ]
        story_sentences.extend(opening_templates[:sentences_per_part['opening']])
        
        # 2. 발전 부분
        for i in range(sentences_per_part['development']):
            if i < len(primary_keywords):
                keyword = primary_keywords[i]
                action = vocab_categories['actions'][i % len(vocab_categories['actions'])] if vocab_categories['actions'] else 'work'
                obj = vocab_categories['objects'][i % len(vocab_categories['objects'])] if vocab_categories['objects'] else keyword
                
                sentence = f"The {character} would {action} with the {obj} because {keyword} was very important."
            else:
                descriptor = vocab_categories['descriptors'][i % len(vocab_categories['descriptors'])] if vocab_categories['descriptors'] else 'good'
                emotion = vocab_categories['emotions'][i % len(vocab_categories['emotions'])] if vocab_categories['emotions'] else 'happy'
                
                sentence = f"This made the {character} feel {emotion} and {descriptor}."
            
            story_sentences.append(sentence)
        
        # 3. 절정
        climax = f"One day, something {vocab_categories['descriptors'][0] if vocab_categories['descriptors'] else 'special'} happened with {primary_keywords[0] if primary_keywords else 'everything'}."
        story_sentences.append(climax)
        
        # 4. 결론
        conclusion = f"In the end, the {character} learned that {', '.join(primary_keywords) if primary_keywords else 'learning'} was the most important thing."
        story_sentences.append(conclusion)
        
        # 문장들을 자연스럽게 연결
        enhanced_story = []
        for i, sentence in enumerate(story_sentences):
            if i > 0 and i < len(story_sentences) - 1:
                # 중간 문장들에 연결어 추가
                connectors = ['Then', 'After that', 'Also', 'However', 'Meanwhile', 'Furthermore']
                if i % 2 == 0 and i > 1:
                    sentence = f"{connectors[i % len(connectors)].lower()}, {sentence.lower()}"
            
            enhanced_story.append(sentence)
        
        return " ".join(enhanced_story)
    
    def _reconstruct_sentence(self, words: List[str]) -> str:
        """
        Reconstruct a grammatically correct sentence from filtered words
        """
        if not words:
            return ""
        
        if len(words) < 3:
            # Too few words for a complete sentence
            return " ".join(words).capitalize() + "."
        
        # Basic sentence patterns
        sentence = ""
        remaining_words = words[:]
        
        # Try to identify sentence components
        subjects = ['he', 'she', 'it', 'they', 'we', 'you', 'i', 'the', 'a', 'an', 'this', 'that']
        verbs = ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 'can', 'could', 
                'do', 'does', 'did', 'go', 'went', 'come', 'came', 'make', 'made', 'take', 'took',
                'see', 'saw', 'get', 'got', 'find', 'found', 'think', 'thought']
        
        # Start with a subject if available
        subject_found = False
        for i, word in enumerate(remaining_words):
            if word.lower() in subjects:
                if word.lower() in ['the', 'a', 'an'] and i < len(remaining_words) - 1:
                    # Article + noun
                    sentence = f"{word} {remaining_words[i+1]}"
                    remaining_words = remaining_words[:i] + remaining_words[i+2:]
                    subject_found = True
                    break
                elif word.lower() in ['he', 'she', 'it', 'they', 'we', 'you', 'i']:
                    sentence = word
                    remaining_words.pop(i)
                    subject_found = True
                    break
        
        if not subject_found and remaining_words:
            # Use first word as subject
            sentence = remaining_words[0]
            remaining_words = remaining_words[1:]
        
        # Add a verb
        verb_found = False
        for i, word in enumerate(remaining_words):
            if word.lower() in verbs:
                sentence += f" {word}"
                remaining_words.pop(i)
                verb_found = True
                break
        
        if not verb_found and remaining_words:
            # Add a default verb
            sentence += " is"
        
        # Add remaining words as object/complement
        if remaining_words:
            sentence += " " + " ".join(remaining_words)
        
        # Capitalize and add period
        sentence = sentence.strip().capitalize()
        if not sentence.endswith('.'):
            sentence += "."
        
        return sentence 