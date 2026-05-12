import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ── client ────────────────────────────────────────────────
supabase_client = create_client(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)


# ── read ──────────────────────────────────────────────────
def load_all_notes() -> list[dict]:
    response = supabase_client.table("notes").select("*").execute()
    return response.data or []


# ── write ─────────────────────────────────────────────────
def save_note_to_db(knowledge: dict, url: str, merge_knowledge_fn) -> dict:
    from datetime import datetime

    today             = datetime.now().strftime("%Y-%m-%d")
    new_article_entry = {
        "url":   url,
        "level": knowledge["level"],
        "date":  today
    }

    # check if topic already exists
    existing = supabase_client.table("notes") \
        .select("*") \
        .eq("topic", knowledge["topic"]) \
        .execute()

    if existing.data:
        print(f"Topic '{knowledge['topic']}' exists. Merging...")
        existing_note = existing.data[0]

        merged = merge_knowledge_fn(existing_note, knowledge)
        existing_note["concepts_covered"] = merged["concepts_covered"]
        existing_note["next_to_cover"]    = merged["next_to_cover"]
        existing_note["level"]            = merged["level"]

        # handle old source_urls format
        if "articles_read" not in existing_note or existing_note["articles_read"] is None:
            old_urls = existing_note.get("source_urls", [])
            existing_note["articles_read"] = [
                {"url": u, "level": "unknown", "date": "unknown"}
                for u in old_urls
            ]

        existing_note.pop("source_urls", None)
        existing_note.pop("source_url",  None)

        already_saved = any(
            a["url"] == url for a in existing_note["articles_read"]
        )
        if not already_saved:
            existing_note["articles_read"].append(new_article_entry)

        supabase_client.table("notes").update({
            "concepts_covered": existing_note["concepts_covered"],
            "next_to_cover":    existing_note["next_to_cover"],
            "level":            existing_note["level"],
            "articles_read":    existing_note["articles_read"],
        }).eq("topic", knowledge["topic"]).execute()

        return existing_note

    else:
        knowledge["articles_read"] = [new_article_entry]
        knowledge.pop("source_urls", None)
        knowledge.pop("source_url",  None)

        supabase_client.table("notes").insert({
            "topic":            knowledge["topic"],
            "category":         knowledge.get("category", "Other"),
            "level":            knowledge["level"],
            "concepts_covered": knowledge.get("concepts_covered", []),
            "next_to_cover":    knowledge.get("next_to_cover", ""),
            "articles_read":    knowledge["articles_read"],
            "summary":          knowledge.get("summary", ""),
        }).execute()

        return knowledge