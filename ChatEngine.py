import requests
import os
import time
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not set! Please add it to environment variables.")

GROQ_RATE_LIMIT_UNTIL = 0


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )


def answer_question_with_groq(query, relevant_chunks, chat_history=None):
    global GROQ_RATE_LIMIT_UNTIL

    now = time.time()
    if now < GROQ_RATE_LIMIT_UNTIL:
        wait_seconds = int(GROQ_RATE_LIMIT_UNTIL - now)
        return (
            f"⏳ Groq rate limit reached. Please wait {wait_seconds} seconds before sending a new request.",
            []
        )

    context_parts = []
    used_chunks = []  
    for i, chunk in enumerate(relevant_chunks[:12], 1):
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "N/A")
        content = chunk["content"]

        context_parts.append(f"[Source: {source} | Page: {page}]\n{content}")

        used_chunks.append({
            "source": source,
            "page": page,
            "content": content
        })

    context = "\n\n---\n\n".join(context_parts)

    conversation_summary = ""
    if chat_history and len(chat_history) > 1:
        recent = chat_history[-8:]
        conv_lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            conv_lines.append(f"{role}: {msg['content']}")
        conversation_summary = "\n".join(conv_lines)

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a highly accurate and professional assistant for the Master Biomedical Engineering (MBE) program at Hochschule Anhalt.
CRITICAL RULES:

- Answer EXCLUSIVELY based on the provided document sources or previous conversation history.
- If the question is a follow-up (e.g., "summarize that", "explain more", "what about X"), use the conversation history FIRST.
- If no relevant information exists: Reply exactly "No sufficient information in the available documents."
- Use the SAME language as the user's question (English or German).
- Be concise, clear, and professional. Use bullet points or numbering when listing items.
- Always cite sources briefly (e.g., "According to SPO MBE 2024, page X...").
- NEVER hallucinate, explain your reasoning, or add external knowledge.
- For summarization requests of entire documents (e.g., module handbook, SPO): Provide a high-level overview including program duration, total credits, main modules/specializations, semester structure, and key regulations, based on extracted information from sources.
- Always use bullet points or numbered lists for summaries.
- Cite multiple pages/sources where possible.
- When asked about Master's thesis registration or regulations, prioritize information from "94_B14_SPO_MBE..." or "Notes_on_final_theses..." documents.
- For module handbook summaries, list key modules, their credits, and semester distribution if available.
- For counting or lists: Be precise and complete.
- If a catalog contains both an outline (e.g., WPM codes) and a detailed list of modules, always prefer the detailed module list when the user asks for module names.
"""
            },
            {
                "role": "user",
                "content": f"""CONVERSATION HISTORY (for follow-ups only):
{conversation_summary if conversation_summary else "No previous conversation"}

DOCUMENT SOURCES:
{context}

CURRENT QUESTION: {query}

ANSWER directly and precisely:"""
            }
        ],
        "temperature": 0.05,
        "max_tokens": 1500,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=data,
            timeout=60
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"].strip(), used_chunks

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            retry_after = e.response.headers.get("Retry-After")
            wait_time = int(retry_after) if retry_after else 60

            GROQ_RATE_LIMIT_UNTIL = time.time() + wait_time

            return (
                f"⛔ Groq rate limit reached.\n"
                f"⏳ Please wait {wait_time} seconds before trying again.",
                []
            )

        return f"❌ HTTP Error: {str(e)}", []

    except Exception as e:
        return f"❌ Error: {str(e)}", []