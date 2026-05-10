import trafilatura
import urllib.request
import json
import os
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(BASE_DIR, "data", "notes")

NOISE_WORDS = {
    "basic", "basics", "intro", "introduction", "advanced",
    "intermediate", "guide", "tutorial", "overview", "fundamentals"
}

CATEGORIES = ["Tech", "Finance", "Career", "Personal", "Other"]

# --- one prompt per category ---
CATEGORY_PROMPTS = {

    "Tech": """
You are extracting knowledge from a TECH article.
Focus on: concepts, workflows, code insights, tool comparisons.

Rules per type:
- CONCEPT:    "Vector Embeddings — converting text into numbers for similarity search"
- WORKFLOW:   "RAG Pipeline — query → embed → retrieve chunks → generate answer"
- CODE:       "Trafilatura usage — fetch_url() gets HTML, extract() returns clean text"
- COMPARISON: "REST vs GraphQL — REST is simpler, GraphQL gives flexible queries"

Aim for 5-8 concepts. Capture workflows as step1 → step2 → step3.
""",

    "Finance": """
You are extracting knowledge from a FINANCE article.
Focus on: what the instrument/concept is, how it works, benefits, limitations, risks.

Rules per type:
- DEFINITION: "Liquid Fund — a mutual fund that invests in short term debt instruments with high liquidity"
- MECHANISM:  "How SIP works — fixed amount invested monthly regardless of market price, averages cost over time"
- BENEFIT:    "Tax benefit of ELSS — investments up to 1.5L deductible under 80C with 3 year lock-in"
- RISK:       "Liquid fund risk — low but not zero, NAV can fall if underlying bonds default"
- COMPARISON: "FD vs Liquid Fund — FD has fixed returns and lock-in, liquid fund has market returns and instant redemption"

Aim for 4-6 concepts. Always capture risks and limitations if mentioned.
""",

    "Career": """
You are extracting knowledge from a CAREER article.
Focus on: actionable strategies, frameworks, real world advice, mistakes to avoid.

Rules per type:
- STRATEGY:   "Salary negotiation — never give first number, anchor high with market research data"
- FRAMEWORK:  "STAR method — Situation, Task, Action, Result format for interview answers"
- MISTAKE:    "Resume mistake — listing responsibilities not achievements, use numbers to show impact"
- INSIGHT:    "Job search insight — 80% of jobs filled through referrals, not job boards"

Aim for 4-6 concepts. Focus on what someone can DO, not just what they know.
""",

    "Personal": """
You are extracting knowledge from a PERSONAL DEVELOPMENT article.
Focus on: actionable tips, habits, mindset shifts, communication techniques.

Rules per type:
- TIP:        "Active listening — maintain eye contact, nod, summarise what they said before responding"
- HABIT:      "Morning routine — first 30 mins no phone, sets focused tone for the day"
- MINDSET:    "Growth mindset — failures are data not identity, reframe as learning opportunities"
- TECHNIQUE:  "Confidence body language — stand tall, speak slower, pause before answering"

Aim for 4-6 concepts. Focus on things the reader can actually practise.
""",

    "Other": """
You are extracting knowledge from a general article.
Focus on: key ideas, important facts, useful insights, things worth remembering.

Aim for 4-6 concepts. Capture whatever is most useful and memorable.
"""
}


def clean_topic_slug(topic: str) -> str:
    words   = topic.lower().split()
    cleaned = [w for w in words if w not in NOISE_WORDS]
    if not cleaned:
        cleaned = words
    return "_".join(cleaned)


def fetch_article(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            raw_html = response.read().decode("utf-8", errors="ignore")
        text = trafilatura.extract(raw_html)
        if text and len(text) > 200:
            print(f"Strategy 1 worked. Got {len(text)} characters.")
            return text
    except Exception as e:
        print(f"Strategy 1 failed: {e}")

    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False
        )
        if text and len(text) > 200:
            print(f"Strategy 2 worked. Got {len(text)} characters.")
            return text
    except Exception as e:
        print(f"Strategy 2 failed: {e}")

    return None


def detect_category(article_text: str) -> str:
    prompt = f"""
Read the first part of this article and classify it into ONE of these categories:
Tech, Finance, Career, Personal, Other

Rules:
- Tech: programming, tools, frameworks, AI, databases, engineering
- Finance: investing, stocks, mutual funds, tax, budgeting, money
- Career: jobs, interviews, salary, resume, workplace, networking
- Personal: habits, mindset, communication, productivity, relationships
- Other: anything that doesn't fit above

Return ONLY the category name. Nothing else. No explanation.

ARTICLE START:
{article_text[:800]}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=10
    )
    result = response.choices[0].message.content.strip()

    # safety net — if model returns something unexpected
    for cat in CATEGORIES:
        if cat.lower() in result.lower():
            return cat
    return "Other"


def extract_knowledge(article_text: str, category: str = "Tech") -> dict:
    category_prompt = CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["Other"])

    prompt = f"""
You are a personal knowledge tracker helping someone remember what they learned.

{category_prompt}

Return ONLY valid JSON. No explanation. No markdown. No backticks. Raw JSON only.

Use EXACTLY this structure:
{{
  "topic": "core topic only, 1-3 words. Strip qualifier words like Basic, Intro, Guide. Examples: RAG, Liquid Fund, Salary Negotiation, Active Listening",
  "category": "{category}",
  "concepts_covered": [
    "concept name — rich description"
  ],
  "summary": [
    "point 1: what this article teaches",
    "point 2: what this article teaches",
    "point 3: what this article teaches"
  ],
  "next_to_cover": "ONE specific topic to study next, specific enough to search directly.",
  "level": "beginner or intermediate or advanced"
}}

ARTICLE:
{article_text[:5000]}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


def merge_knowledge(existing_note: dict, new_note: dict) -> dict:
    prompt = f"""
You are a knowledge merging agent.

Merge two notes on: "{existing_note['topic']}"

EXISTING concepts:
{json.dumps(existing_note['concepts_covered'], indent=2)}

NEW concepts:
{json.dumps(new_note['concepts_covered'], indent=2)}

EXISTING next_to_cover: {existing_note['next_to_cover']}
NEW next_to_cover: {new_note['next_to_cover']}
EXISTING level: {existing_note['level']}
NEW level: {new_note['level']}

Rules:
- Same concept = merge into one richer description
- Different concept = keep both
- next_to_cover: pick the most specific one
- level: always pick the more advanced level

Return ONLY valid JSON:
{{
  "concepts_covered": ["concept name — merged rich description"],
  "next_to_cover": "single most useful next topic",
  "level": "beginner or intermediate or advanced"
}}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


def save_note(knowledge: dict, url: str) -> dict:
    os.makedirs(NOTES_DIR, exist_ok=True)

    category   = knowledge.get("category", "Other")
    topic_slug = clean_topic_slug(knowledge["topic"])

    # store by category/topic so notes are organized
    category_dir = os.path.join(NOTES_DIR, category.lower())
    os.makedirs(category_dir, exist_ok=True)
    filepath = os.path.join(category_dir, f"{topic_slug}.json")

    today             = datetime.now().strftime("%Y-%m-%d")
    new_article_entry = {
        "url":   url,
        "level": knowledge["level"],
        "date":  today
    }

    if os.path.exists(filepath):
        print(f"Topic '{knowledge['topic']}' exists. Merging...")
        with open(filepath, "r", encoding="utf-8") as f:
            existing_note = json.load(f)

        merged = merge_knowledge(existing_note, knowledge)
        existing_note["concepts_covered"] = merged["concepts_covered"]
        existing_note["next_to_cover"]    = merged["next_to_cover"]
        existing_note["level"]            = merged["level"]

        if "articles_read" not in existing_note:
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

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing_note, f, indent=2)

        return existing_note

    else:
        knowledge["articles_read"] = [new_article_entry]
        knowledge.pop("source_urls", None)
        knowledge.pop("source_url",  None)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(knowledge, f, indent=2)

        return knowledge