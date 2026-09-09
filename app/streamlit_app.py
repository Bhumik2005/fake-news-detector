import os
import requests
import streamlit as st
from duckduckgo_search import DDGS
from google import genai

# Page Config
st.set_page_config(page_title="Global Fact Check Engine", page_icon="🔍", layout="wide")

st.title("🌐 Global News & Fact-Checking Engine")
st.caption("Production-grade verification engine using Live Search & AI reasoning.")

# Get Gemini API key from Streamlit secrets or environment
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

# Step 1: Query Google Fact Check API
def query_factcheck_api(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={st.secrets.get('GOOGLE_FACTCHECK_KEY', '')}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("claims", [])
    except Exception:
        pass
    return []

# Step 2: Live Web Search
def search_live_news(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(f"{query} news fact check", max_results=5))
    return results

# Step 3: LLM Synthesis with Gemini
def analyze_with_ai(claim, search_results):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    context = "\n".join([f"- Title: {r['title']}\n  Snippet: {r['body']}\n  URL: {r['href']}" for r in search_results])
    
    prompt = f"""
    You are a professional global news fact-checker. Evaluate the following claim using the provided live search evidence.

    Claim: "{claim}"

    Search Evidence:
    {context}

    Provide your evaluation in the following structure:
    1. Verdict: (TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIED)
    2. Confidence Score: (0-100%)
    3. Summary Explanation: (3-4 concise sentences detailing why)
    4. Key Sources Referenced: (Bullet list of URLs)
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.text

# User Input Interface
claim_input = st.text_area("Enter a headline, news claim, or article snippet:", placeholder="e.g., NASA announced discovery of liquid water on Mars today...")

if st.button("Analyze & Verify", type="primary"):
    if not claim_input.strip():
        st.warning("Please enter a claim to analyze.")
    else:
        with st.spinner("Step 1: Searching global fact-checking databases..."):
            fact_checks = query_factcheck_api(claim_input)
            
        if fact_checks:
            st.success("Found existing fact-checks from verified organizations:")
            for claim in fact_checks[:3]:
                st.write(f"**Claim:** {claim.get('text')}")
                for review in claim.get('claimReview', []):
                    st.info(f"**Publisher:** {review.get('publisher', {}).get('name')} | **Rating:** {review.get('textualRating')} | [Read Article]({review.get('url')})")
        else:
            with st.spinner("Step 2 & 3: Cross-referencing live global sources with AI..."):
                search_results = search_live_news(claim_input)
                if not search_results:
                    st.error("No live web results found to verify this claim.")
                elif not GEMINI_API_KEY:
                    st.error("Please configure GEMINI_API_KEY in `.streamlit/secrets.toml` to enable AI analysis.")
                else:
                    verdict = analyze_with_ai(claim_input, search_results)
                    st.subheader("Analysis Verdict & Evidence")
                    st.markdown(verdict)