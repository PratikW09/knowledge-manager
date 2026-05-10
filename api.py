from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import json
import os

from extractor import (
    fetch_article, detect_category,
    extract_knowledge, save_note, NOTES_DIR
)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class URLInput(BaseModel):
    url: str

class SaveInput(BaseModel):
    url:      str
    category: str  # confirmed by user

class ChatInput(BaseModel):
    question: str
    history:  list[dict] = []


def load_all_notes() -> list[dict]:
    notes = []
    if not os.path.exists(NOTES_DIR):
        return notes

    # walk all category subfolders
    for root, dirs, files in os.walk(NOTES_DIR):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                    notes.append(json.load(f))
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    return notes


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


@app.get("/")
def root():
    return {"status": "running"}


# step 1 — fetch article + detect category
@app.post("/detect")
def detect(data: URLInput):
    article_text = fetch_article(data.url)

    if not article_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract content. Try dev.to, freecodecamp, or realpython."
        )

    category = detect_category(article_text)

    # store article text temporarily in memory so
    # /save doesn't fetch again — simple and fast
    app.state.article_cache = {
        "url":  data.url,
        "text": article_text
    }

    return {
        "detected_category": category,
        "url": data.url
    }


# step 2 — user confirmed category, now extract and save
@app.post("/save")
def save_article(data: SaveInput):
    cache = getattr(app.state, "article_cache", None)

    # use cached text if available, else re-fetch
    if cache and cache["url"] == data.url:
        article_text = cache["text"]
    else:
        article_text = fetch_article(data.url)

    if not article_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract content."
        )

    knowledge  = extract_knowledge(article_text, data.category)
    final_note = save_note(knowledge, data.url)

    return {
        "status":        "saved",
        "topic":         final_note["topic"],
        "category":      final_note.get("category", data.category),
        "level":         final_note["level"],
        "articles_read": len(final_note.get("articles_read", []))
    }


@app.get("/notes")
def get_all_notes():
    notes  = load_all_notes()
    topics = []

    for note in notes:
        topics.append({
            "topic":            note.get("topic", ""),
            "category":         note.get("category", "Other"),
            "level":            note.get("level", "beginner"),
            "concepts_covered": note.get("concepts_covered", []),
            "next_to_cover":    note.get("next_to_cover", ""),
            "articles_read":    note.get("articles_read", []),
            "summary":          note.get("summary", []),
        })

    total_articles = sum(len(t["articles_read"]) for t in topics)
    total_concepts = sum(len(t["concepts_covered"]) for t in topics)

    journey = []
    for t in topics:
        for article in t["articles_read"]:
            journey.append({
                "date":     article.get("date",  "unknown"),
                "topic":    t["topic"],
                "category": t["category"],
                "level":    article.get("level", "unknown"),
                "url":      article.get("url",   ""),
            })
    journey.sort(key=lambda x: x["date"], reverse=True)

    return {
        "topics":  topics,
        "journey": journey,
        "stats": {
            "total_topics":   len(topics),
            "total_articles": total_articles,
            "total_concepts": total_concepts,
        }
    }


@app.post("/chat")
def chat(data: ChatInput):
    notes   = load_all_notes()
    context = build_knowledge_context(notes)

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
    for msg in data.history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": data.question})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=400
    )

    return {"answer": response.choices[0].message.content.strip()}