import os
import re
import sqlite3
from typing import List, Literal
from PIL import Image
import streamlit as st
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. Page Configuration & UI Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Global Fact-Checking Engine",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Global News & Fact-Checking Engine")
st.caption("Production-grade verification engine using Live Search, Vision & AI reasoning.")

# -----------------------------------------------------------------------------
# 2. Database Initialization (SQLite Feedback Logging)
# -----------------------------------------------------------------------------
def init_db():
    """Initializes a local SQLite database to log user feedback and edge cases."""
    conn = sqlite3.connect("feedback_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            claim TEXT,
            verdict TEXT,
            confidence_score INTEGER,
            user_rating INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_feedback(claim: str, verdict: str, confidence: int, rating: int):
    """Saves user feedback rating (0 = thumbs down, 1 = thumbs up) to SQLite."""
    conn = sqlite3.connect("feedback_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (claim, verdict, confidence_score, user_rating)
        VALUES (?, ?, ?, ?)
    """, (claim, verdict, confidence, rating))
    conn.commit()
    conn.close()

init_db()

# -----------------------------------------------------------------------------
# 3. Safe API Key Initialization
# -----------------------------------------------------------------------------
GEMINI_API_KEY = None

try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Please configure GEMINI_API_KEY in .streamlit/secrets.toml to enable AI analysis.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------------------------------------------------------
# 4. Pydantic Schema for Guaranteed Structured Output
# -----------------------------------------------------------------------------
class FactCheckReport(BaseModel):
    verdict: Literal["REAL NEWS", "FAKE NEWS", "MISLEADING", "UNVERIFIABLE"] = Field(
        description="The strict classification label for the news claim."
    )
    confidence_score: int = Field(
        description="Confidence level in percentage from 0 to 100.",
        ge=0,
        le=100
    )
    summary: str = Field(
        description="Executive summary explaining the reasoning behind the verdict."
    )
    key_findings: List[str] = Field(
        description="List of key bulleted facts supporting or debunking the claim."
    )

# -----------------------------------------------------------------------------
# 5. Helper Functions (RAG, Vision & Search)
# -----------------------------------------------------------------------------
def sanitize_search_query(text: str) -> str:
    """Removes conversational fluff to generate cleaner search terms for DuckDuckGo."""
    query = text.lower()
    filler_words = [
        "is the", "are there", "was there", "still going on", "tell me if",
        "is it true that", "did", "does", "what happened to", "between"
    ]
    for word in filler_words:
        query = query.replace(word, " ")
    
    query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    query = " ".join(query.split())
    return query if len(query) > 3 else text

def fetch_live_context_and_sources(query: str, max_results: int = 5):
    """Fetches real-time web search snippets and structured source links using DuckDuckGo."""
    clean_query = sanitize_search_query(query) if query else ""
    if not clean_query:
        return "", []
        
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(clean_query, max_results=max_results))
            if not results and query:
                results = list(ddgs.text(query, max_results=max_results))
                
            if not results:
                return "", []
            
            context_blocks = []
            structured_sources = []
            
            for idx, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                body = r.get("body", "No Content")
                href = r.get("href", "#")
                
                context_blocks.append(f"Source [{idx}]: {title}\nURL: {href}\nSnippet: {body}\n")
                structured_sources.append({"title": title, "url": href, "snippet": body})
            
            formatted_context = "\n---\n".join(context_blocks)
            return formatted_context, structured_sources
    except Exception as e:
        st.warning(f"Web search warning: {e}")
        return "", []

def verify_claim_with_gemini(claim_text: str, image: Image.Image | None, context: str) -> FactCheckReport:
    """Uses Gemini 3.6 Flash with Structured Outputs to analyze text and image claims."""
    prompt = f"""
    You are an expert investigative fact-checker and machine learning classification model.
    Analyze the user claim (and/or uploaded screenshot) against the provided live web search context.

    USER TEXT CLAIM:
    "{claim_text if claim_text else 'Analyze the provided image for fake news/misinformation.'}"

    LIVE RETRIEVED WEB CONTEXT:
    {context if context else "No relevant live context retrieved."}

    INSTRUCTIONS:
    1. If an image is provided, extract any embedded text or claim from the image and verify it.
    2. Determine veracity strictly using one of the allowed verdict options.
    3. Calculate an accurate confidence score based on available supporting evidence.
    """
    
    contents = [prompt]
    if image:
        contents.append(image)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=FactCheckReport,
        )
    )
    return response.parsed

# -----------------------------------------------------------------------------
# 6. Main Application UI & Logic
# -----------------------------------------------------------------------------
user_text = st.text_area(
    "Enter a headline, news claim, or article snippet:",
    placeholder="e.g., Israel and Iran conflict updates",
    height=100
)

uploaded_file = st.file_uploader(
    "Optionally upload a headline screenshot or image:",
    type=["jpg", "jpeg", "png"]
)

pil_image = None
if uploaded_file:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded Image Context", use_column_width=True)

if st.button("Analyze & Verify", type="primary"):
    if not user_text.strip() and not pil_image:
        st.warning("Please enter a text claim or upload an image to analyze.")
    else:
        with st.spinner("Searching global databases, running OCR/Vision, and analyzing sources..."):
            search_query = user_text if user_text else "breaking news verification"
            search_context, sources = fetch_live_context_and_sources(search_query)
            
            try:
                report: FactCheckReport = verify_claim_with_gemini(user_text, pil_image, search_context)
                
                # Cache recent report state in Streamlit session memory for feedback logging
                st.session_state["last_claim"] = user_text if user_text else "Uploaded Image Claim"
                st.session_state["last_report"] = report
                st.session_state["sources"] = sources
                st.session_state["search_context"] = search_context
                
            except Exception as e:
                st.error(f"Error during AI analysis: {e}")

# Render Results & Feedback section if analysis exists in session state
if "last_report" in st.session_state:
    report: FactCheckReport = st.session_state["last_report"]
    sources = st.session_state["sources"]
    search_context = st.session_state["search_context"]
    
    st.divider()
    st.subheader("Verification Report")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if report.verdict == "REAL NEWS":
            st.success(f"### Verdict: {report.verdict}")
        elif report.verdict == "FAKE NEWS":
            st.error(f"### Verdict: {report.verdict}")
        elif report.verdict == "MISLEADING":
            st.warning(f"### Verdict: {report.verdict}")
        else:
            st.info(f"### Verdict: {report.verdict}")
            
    with col2:
        st.metric(label="Model Confidence Score", value=f"{report.confidence_score}%")
    
    st.markdown("#### Summary")
    st.write(report.summary)
    
    st.markdown("#### Key Findings")
    for fact in report.key_findings:
        st.markdown(f"- {fact}")
    
    # Clickable Sources
    if sources:
        st.divider()
        st.subheader("🔗 Verified News Sources & References")
        for idx, src in enumerate(sources, 1):
            st.markdown(f"**{idx}. [{src['title']}]({src['url']})**")
            st.caption(src['snippet'])
            st.write("")

    # --- User Feedback & Data Collection Loop ---
    st.divider()
    st.caption("Was this verification report accurate?")
    feedback_rating = st.feedback("thumbs")
    
    if feedback_rating is not None:
        log_feedback(
            claim=st.session_state["last_claim"],
            verdict=report.verdict,
            confidence=report.confidence_score,
            rating=feedback_rating
        )
        st.toast("Thank you! Your feedback has been logged to the database.", icon="✅")

    # Raw Search Drawer
    if search_context:
        with st.expander("View Raw Search Data"):
            st.text(search_context)