import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"


# ── context builder ───────────────────────────────────────
def build_knowledge_context(notes: list[dict]) -> str:
    if not notes:
        return "This user has no notes yet."

    lines = ["USER KNOWLEDGE BASE:\n"]
    for note in notes:
        topic    = note.get("topic", "")
        level    = note.get("level", "beginner")
        category = note.get("category", "Other")
        concepts = note.get("concepts_covered", [])
        next_up  = note.get("next_to_cover", "")
        articles = len(note.get("articles_read", []))

        lines.append(f"[{category}] {topic} — {level} — {articles} article(s)")
        for c in concepts:
            lines.append(f"  • {c}")
        lines.append(f"  → not yet covered: {next_up}")
        lines.append("")

    return "\n".join(lines)


# ── chat ──────────────────────────────────────────────────
def ask_groq(question: str, history: list[dict], notes: list[dict]) -> str:
    context       = build_knowledge_context(notes)
    system_prompt = f"""
You are the user's personal knowledge assistant — sharp, minimal, direct.

Their knowledge spans multiple areas — tech, finance, career, personal growth.
Answer using ONLY what's in their notes unless they explicitly ask otherwise.

{context}

How to answer:
COVERAGE ("what have I covered / what do I know about X"):
→ bullet list only. Category, topic, level, key concepts. No prose.

GAP ("what am I missing / what should I learn next"):
→ one line per topic. Specific. No fluff.

EXPLANATION ("explain X"):
→ 3-4 lines max. Connect to what they already know.

Rules:
- Never say "Great question" or "As an AI"
- Never write paragraphs when bullets work
- Max 400 tokens per response
- If topic not in notes: "You haven't covered this yet. Want me to explain it?"
"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=400
    )

    return response.choices[0].message.content.strip()