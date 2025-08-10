from typing import List, Optional
import os

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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


def _get_llm():
    """Return AzureChatOpenAI if AOAI_* env vars are set, else ChatOpenAI."""
    if os.getenv("AOAI_API_KEY") and os.getenv("AOAI_ENDPOINT") and os.getenv("AOAI_API_VERSION") and os.getenv("AOAI_DEPLOY_GPT4O"):
        return AzureChatOpenAI(
            api_key=os.getenv("AOAI_API_KEY"),
            azure_endpoint=os.getenv("AOAI_ENDPOINT"),
            api_version=os.getenv("AOAI_API_VERSION"),
            deployment_name=os.getenv("AOAI_DEPLOY_GPT4O"),
            temperature=0.7,
        )
    # Fallback to OpenAI
    return ChatOpenAI(model="gpt-4", temperature=0.7)


def _build_vocabulary_instruction(available_vocabulary: Optional[List[str]]) -> str:
    if not available_vocabulary:
        return "No strict vocabulary restriction. Prefer simple, clear wording and maintain grammatical correctness."
    vocab_sample = available_vocabulary[:100]
    return (
        "CRITICAL VOCABULARY RESTRICTION - FOLLOW EXACTLY.\n\n"
        f"1) RAG Vocabulary (sample): {', '.join(vocab_sample)}\n"
        f"2) Essential Grammar Words: {', '.join(sorted(essential_words))}\n\n"
        "Rules:\n"
        "- Use ONLY words from the allowed lists above for content words.\n"
        "- If a needed word is not allowed, rephrase using RAG vocabulary.\n"
        "- Keep grammar perfect and sentences meaningful.\n"
    )


def generate_story_langchain(
    keywords: List[str],
    context_documents: Optional[List[str]] = None,
    story_length: str = "medium",
    available_vocabulary: Optional[List[str]] = None,
) -> str:
    """Generate a story using LangChain prompt + LLM with optional vocabulary restriction."""
    length_settings = {
        "short": {"sentences": "3-5 sentences", "words": "100-200 words"},
        "medium": {"sentences": "6-10 sentences", "words": "200-400 words"},
        "long": {"sentences": "10-15 sentences", "words": "400-800 words"},
    }
    settings = length_settings.get(story_length, length_settings["medium"])

    context_text = ""
    if context_documents:
        limited_docs = []
        for doc in context_documents[:3]:
            limited_docs.append(doc[:200] + ("..." if len(doc) > 200 else ""))
        context_text = "\n".join(limited_docs)

    vocabulary_instruction = _build_vocabulary_instruction(available_vocabulary)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful story writer."),
        (
            "user",
            (
                "Create an engaging English story.\n\n"
                "PRIMARY KEYWORDS (use prominently): {primary}\n"
                "SECONDARY KEYWORDS (optional): {secondary}\n"
                "STORY LENGTH: {sentences} (~{words})\n\n"
                "CONTEXT INFORMATION:\n{context}\n\n"
                "VOCABULARY INSTRUCTION:\n{vocab_instruction}\n\n"
                "STORY REQUIREMENTS:\n"
                "- Write exactly {sentences} with approximately {words}.\n"
                "- Clear beginning, middle, and end; logical and grammatical.\n"
                "- Smooth transitions; proper punctuation and capitalization.\n"
                "Write the story now."
            ),
        ),
    ])

    primary_keywords = keywords[:3]
    secondary_keywords = keywords[3:] if len(keywords) > 3 else []

    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()
    story = chain.invoke({
        "primary": ", ".join(primary_keywords),
        "secondary": ", ".join(secondary_keywords) if secondary_keywords else "None",
        "sentences": settings["sentences"],
        "words": settings["words"],
        "context": context_text or "Use creativity to build context around the keywords.",
        "vocab_instruction": vocabulary_instruction,
    })
    return story.strip() 