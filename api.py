from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from Models   import URLInput, SaveInput, ChatInput
from Database import load_all_notes, save_note_to_db
from Chat     import ask_groq
from extractor import (
    fetch_article, detect_category,
    extract_knowledge, merge_knowledge
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── health ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running"}


# ── step 1: fetch article + detect category ───────────────
@app.post("/detect")
def detect(data: URLInput):
    article_text = fetch_article(data.url)

    if not article_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract content. Try dev.to, freecodecamp, or realpython."
        )

    category = detect_category(article_text)

    # cache article text in memory so /save doesn't re-fetch
    app.state.article_cache = {
        "url":  data.url,
        "text": article_text
    }

    return {
        "detected_category": category,
        "url": data.url
    }


# ── step 2: user confirmed category → extract + save ──────
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
    final_note = save_note_to_db(knowledge, data.url, merge_knowledge)

    return {
        "status":        "saved",
        "topic":         final_note["topic"],
        "category":      final_note.get("category", data.category),
        "level":         final_note["level"],
        "articles_read": len(final_note.get("articles_read", []))
    }


# ── get all notes + journey + stats ───────────────────────
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
            "summary":          note.get("summary", ""),
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


# ── chat with knowledge assistant ─────────────────────────
@app.post("/chat")
def chat(data: ChatInput):
    notes  = load_all_notes()
    answer = ask_groq(data.question, data.history, notes)
    return {"answer": answer}