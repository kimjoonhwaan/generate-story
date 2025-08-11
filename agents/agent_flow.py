from __future__ import annotations
from typing import TypedDict, List, Optional, Dict, Any
import os
from functools import partial
import json
import re
import time

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool

from lc_pipeline import generate_story_langchain
from story_generator import StoryGenerator


class State(TypedDict):
    keywords: str
    length: str
    use_rag_only: bool
    allowed_vocab: List[str]
    context: List[str]
    story: str
    critique: str
    tries: int
    ok: bool
    logs: List[str]


def _log(state: State, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    state.setdefault("logs", []).append(line)
    try:
        print(f"[MultiAgent] {line}", flush=True)
    except Exception:
        pass


def get_llm():
    if os.getenv("AOAI_API_KEY") and os.getenv("AOAI_ENDPOINT") and os.getenv("AOAI_API_VERSION") and os.getenv("AOAI_DEPLOY_GPT4O"):
        return AzureChatOpenAI(
            api_key=os.getenv("AOAI_API_KEY"),
            azure_endpoint=os.getenv("AOAI_ENDPOINT"),
            api_version=os.getenv("AOAI_API_VERSION"),
            deployment_name=os.getenv("AOAI_DEPLOY_GPT4O"),
            temperature=0.2,
        )
    return ChatOpenAI(model="gpt-4", temperature=0.2)


# --- Tool funcs (will be wrapped with Tool and capture rag_system) ---

# 상단 import 보강
import json, re
from functools import partial

# 안전 파서
def _safe_parse(params: str) -> dict:
    if not isinstance(params, str) or not params.strip():
        return {}
    try:
        return json.loads(params)
    except Exception:
        s = re.sub(r"^```(json)?|```$", "", params.strip(), flags=re.IGNORECASE|re.MULTILINE).strip()
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # "query: something" 같은 패턴 보정
        m2 = re.search(r"query\s*:\s*(.+)", s, flags=re.IGNORECASE)
        return {"query": m2.group(1).strip()} if m2 else {}

# 툴 함수들 단일 문자열(JSON) 입력 → 안전 파서 적용
def _retrieve_docs(params: str, rag_system=None, default_query: str = "") -> List[str]:
    # 1) 기본 쿼리 우선
    query = (default_query or "").strip()

    # 2) 기본 쿼리가 비었을 때만 입력(JSON)에서 보완
    if not query:
        data = _safe_parse(params)
        query = (data.get("query") or "").strip()
        if not query and data.get("keywords"):
            query = str(data["keywords"]).split(",")[0].strip()

    if not query:
        raise ValueError("retrieve_docs: empty query")

    n = 3
    try:
        data = _safe_parse(params)
        if "n" in data:
            n = int(data.get("n", 3))
    except Exception:
        pass

    results = rag_system.vector_db.search(query, n_results=n) if rag_system else []
    return [r["document"] for r in results]

def _generate_draft(params: str) -> str:
    data = _safe_parse(params)
    keywords = data.get("keywords", "")
    context = data.get("context", [])
    allowed_vocab = data.get("allowed_vocab", [])
    length = data.get("length", "medium")
    kws = [k.strip() for k in str(keywords).split(",") if k.strip()]
    return generate_story_langchain(kws, context, length, allowed_vocab or None)

def _vocab_analysis(params: str) -> str:
    data = _safe_parse(params)
    story = data.get("story", "")
    allowed_vocab = data.get("allowed_vocab", [])
    sg = StoryGenerator(use_openai=False)
    return sg._annotate_non_rag_words(story, allowed_vocab)

def _revise_with_constraints(params: str) -> str:
    data = _safe_parse(params)
    story = data.get("story", "")
    critique = data.get("critique", "")
    allowed_vocab = data.get("allowed_vocab", [])
    length = data.get("length", "medium")
    context = [story, f"Critique for revision: {critique}"]
    return generate_story_langchain([], context, length, allowed_vocab or None)


# 2) Tool 정의: rag_system 주입 및 설명에 JSON 명세 강조
from functools import partial
from langchain.tools import Tool

def _generate_draft_with_defaults(params: str, default_keywords: str, default_context: list, default_allowed: list, default_length: str) -> str:
    data = _safe_parse(params)
    keywords = data.get("keywords", default_keywords)
    context = data.get("context", default_context)
    allowed = data.get("allowed_vocab", default_allowed)
    length  = data.get("length", default_length)
    kws = [k.strip() for k in str(keywords).split(",") if k.strip()]
    return generate_story_langchain(kws, context or [], length, allowed or None)

def _vocab_analysis_with_defaults(params: str, default_story: str = "", default_allowed: list = None) -> str:
    data = _safe_parse(params)
    story  = data.get("story", default_story)
    allowed = data.get("allowed_vocab", default_allowed or [])
    sg = StoryGenerator(use_openai=False)
    return sg._annotate_non_rag_words(story, allowed or [])

def _make_tools(default_keywords: str, default_context: list, default_allowed: list, default_length: str) -> list[Tool]:
    return [
        Tool(
            name="generate_draft",
            func=partial(
                _generate_draft_with_defaults,
                default_keywords=default_keywords,
                default_context=default_context,
                default_allowed=default_allowed,
                default_length=default_length,
            ),
            description='Input JSON: {"keywords":"a, b", "context":["..."], "allowed_vocab":["..."], "length":"short|medium|long"} (all optional)'
        ),
        Tool(
            name="vocab_analysis",
            func=partial(
                _vocab_analysis_with_defaults,
                default_story="",  # 호출 시 전달되지 않으면 호출자가 넣은 story를 그대로 사용
                default_allowed=default_allowed,
            ),
            description='Input JSON: {"story":"...", "allowed_vocab":["..."]} (both optional)'
        ),
    ]


def node_retrieve(state: State, rag_system) -> State:
    _log(state, "Retrieve: start")
    # Derive a safe default query from the first non-empty keyword
    raw = str(state.get("keywords", ""))
    first_kw = next((k.strip() for k in raw.split(",") if k.strip()), "")
    if first_kw:
        try:
            results = rag_system.vector_db.search(first_kw, n_results=3)
            state["context"] = [r["document"] for r in results]
            _log(state, f"Retrieve: ok (query='{first_kw}', docs={len(state['context'])})")
        except Exception as e:
            state["context"] = []
            _log(state, f"Retrieve: failed ({e}) — continue without context")
    else:
        state["context"] = []
        _log(state, "Retrieve: no keyword — skipping context")

    # If RAG-only is requested, (re)compute an adequate allowed vocabulary
    if state.get("use_rag_only"):
        try:
            words = rag_system.vector_db.get_filtered_vocabulary(state.get("keywords", ""), state.get("context", []))
            # Fallback to full vocabulary if too sparse
            if not words or len(words) < 30:
                words = rag_system.vector_db.get_vocabulary()
            state["allowed_vocab"] = words
            _log(state, f"Vocab: prepared (size={len(words)})")
        except Exception as e:
            # As a last resort, use full vocabulary to avoid empty constraints
            try:
                words = rag_system.vector_db.get_vocabulary()
                state["allowed_vocab"] = words
                _log(state, f"Vocab: fallback to full (size={len(words)})")
            except Exception as e2:
                state["allowed_vocab"] = []
                _log(state, f"Vocab: failed to prepare ({e2})")

    return state


def _make_tool_agent(default_keywords: str, default_context: list, default_allowed: list, default_length: str) -> AgentExecutor:
    tools = _make_tools(default_keywords, default_context, default_allowed, default_length)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful ReAct tool agent. You have these tools:\n{tools}\n"
         "Use only these tool names: {tool_names}.\n"
         "Never ask the user any questions. If any information is missing, use provided defaults.\n"
         "Always call tools with valid JSON (fields are optional; defaults are pre-bound).\n"
         "STRICT FORMAT (follow exactly, including newlines):\n"
         "Thought: <your reasoning>\n"
         "Action: <tool_name>\n"
         "Action Input: <JSON only>\n"
         "Observation: <tool result>\n"
         "... (repeat Thought/Action/Action Input/Observation as needed) ...\n"
         "When done, output:\n"
         "Final Answer: <final story or analysis>"
        ),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ])
    llm = get_llm()
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


def node_generate(state: State, rag_system) -> State:
    default_keywords = str(state.get("keywords",""))
    default_context  = state.get("context", [])
    default_allowed  = state.get("allowed_vocab", []) if state.get("use_rag_only") else []
    default_length   = state.get("length", "medium")

    agent = _make_tool_agent(default_keywords, default_context, default_allowed, default_length)

    instruction = (
        f"Generate a {default_length} story.\n"
        f"Keywords: {default_keywords}\n"
        f"Use defaults for any missing inputs. Do not ask the user anything.\n\n"
        "Plan and execute exactly these steps using tools:\n"
        "1) Thought: Prepare to generate draft.\n"
        "Action: generate_draft\n"
        f"Action Input: {{\"keywords\": \"{default_keywords}\", \"context\": {json.dumps(default_context)}, \"allowed_vocab\": {json.dumps(default_allowed)}, \"length\": \"{default_length}\"}}\n"
        "2) Thought: Analyze vocabulary usage.\n"
        "Action: vocab_analysis\n"
        "Action Input: {\"story\": <the generated story from previous Observation>, \"allowed_vocab\": <same allowed_vocab> }\n"
        "Finally, provide:\n"
        "Final Answer: <the fully generated story with appended analysis block>\n"
    )
    _log(state, f"Generate: start (len(context)={len(default_context)}, allowed={len(default_allowed)})")
    t0 = time.time()
    out = _make_tool_agent(default_keywords, default_context, default_allowed, default_length).invoke({"input": instruction})
    dt = time.time() - t0
    text = out.get("output") if isinstance(out, dict) else str(out)
    state["story"] = text
    _log(state, f"Generate: done in {dt:.1f}s")
    return state


def _judge_ok(story: str, allowed: List[str]) -> bool:
    import re as _re
    words = _re.findall(r"[a-zA-Z]+", story.lower())
    essential = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might',
        'must', 'shall', 'and', 'or', 'but', 'so', 'if', 'when', 'where', 'what',
        'who', 'how', 'why', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from',
        'about', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this',
        'that', 'these', 'those', 'here', 'there', 'now', 'then'
    }
    if len(words) < 80:
        return False
    content = [w for w in words if w not in essential]
    if allowed and content:
        rate = sum(w in set(allowed) for w in content) / max(len(content), 1)
        if rate < 0.6:
            return False
    if words.count("thing") > len(words) * 0.05:
        return False
    return True


def node_evaluate(state: State) -> State:
    _log(state, "Evaluate: start")
    state["ok"] = _judge_ok(state["story"], state.get("allowed_vocab") or [])
    state["critique"] = (
        "Increase RAG vocabulary usage to >=60%, avoid generic words, and ensure coherent flow."
        if not state["ok"] else ""
    )
    _log(state, f"Evaluate: ok={state['ok']}")
    return state


def node_revise(state: State, rag_system) -> State:
    if state["ok"]:
        _log(state, "Revise: skipped (already ok)")
        return state
    _log(state, "Revise: start")
    out = _make_tool_agent(state["keywords"], state["context"], state["allowed_vocab"], state["length"]).invoke({
        "input": "Revise the story with constraints, then run vocab_analysis.",
        "story": state["story"],
        "critique": state["critique"],
        "allowed_vocab": state["allowed_vocab"],
        "length": state["length"],
    })
    text = out.get("output") if isinstance(out, dict) else str(out)
    state["story"] = text
    state["tries"] = state.get("tries", 0) + 1
    _log(state, f"Revise: attempt {state['tries']} done")
    return state


def _decide(state: State):
    return END if state["ok"] or state.get("tries", 0) >= 2 else "generate"


def compile_app(rag_system):
    g = StateGraph(State)
    g.add_node("retrieve", lambda s: node_retrieve(s, rag_system))
    g.add_node("generate", lambda s: node_generate(s, rag_system))
    g.add_node("evaluate", node_evaluate)
    g.add_node("revise", lambda s: node_revise(s, rag_system))
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "evaluate")
    g.add_edge("evaluate", "revise")
    g.add_conditional_edges("revise", _decide, {"generate": "generate", END: END})
    g.set_entry_point("retrieve")
    return g.compile()


def run_multi_agent_flow(rag_system, keywords: str, length: str, allowed_vocab: Optional[List[str]]) -> Dict[str, Any]:
    app = compile_app(rag_system)
    init: State = {
        "keywords": keywords,
        "length": length,
        "use_rag_only": bool(allowed_vocab),
        "allowed_vocab": allowed_vocab or [],
        "context": [],
        "story": "",
        "critique": "",
        "tries": 0,
        "ok": False,
        "logs": [],
    }
    final = app.invoke(init)
    return final 

