# 🌐 Global News & Multimodal Fact-Checking Engine

An end-to-end, production-grade Retrieval-Augmented Generation (RAG) system built to verify news headlines, textual claims, and uploaded screenshots in real time.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)
![Gemini](https://img.shields.io/badge/Google%20Gemini-3.6%20Flash-orange.svg)

## 📌 Architecture Highlights
- **Real-Time Web RAG:** Fetches live context snippets via DuckDuckGo API with automated query sanitization to strip conversational fluff.
- **Type-Safe Verification:** Enforces deterministic JSON outputs using Pydantic schemas (`FactCheckReport`) to dynamically render color-coded verdict badges and confidence metrics.
- **Multimodal Capabilities:** Supports direct image uploads (news screenshots, viral posts) powered by Gemini's native vision OCR.
- **Active Learning Data Loop:** Captures user thumbs-up/down ratings into a persistent local SQLite database (`feedback_logs.db`) for future model auditing and prompt refinement.

## 📁 Repository Structure
```text
fake-news-detector/
├── app/
│   └── streamlit_app.py     # Main Streamlit web application & pipeline logic
├── .streamlit/
│   └── secrets.toml         # Local secrets file (API keys - gitignored)
├── feedback_logs.db         # Auto-generated SQLite feedback database
├── requirements.txt         # Project dependencies
├── .gitignore               # Excluded files
└── README.md                # Documentation
```

🚀 Quickstart Guide
1. Clone the repository & setup environment
   ```Bash
   git clone [https://github.com/your-username/fake-news-detector.git](https://github.com/your-username/fake-news-detector.git)
   cd fake-news-detector

   # Create and activate virtual environment
   python -m venv env
   source env/bin/activate  # On Windows: & "env/Scripts/Activate.ps1"
   ```
2. Install dependencies
      ```Bash
         pip install -r requirements.txt
      ```
3. Configure API Key
      Create .streamlit/secrets.toml in the project root:
      ```Ini, TOML
         GEMINI_API_KEY = "your-google-gemini-api-key-here"
      ```
4. Run the Streamlit Application
      ```Bash
          streamlit run app/streamlit_app.py
      ```
   📊 Verification Schema
The model strictly adheres to the following structured verdict categories:

. REAL NEWS - Claim is confirmed supported by verified sources.

. FAKE NEWS - Claim is explicitly refuted by credible source data.

. MISLEADING - Claim contains partially true elements presented out of context.

. UNVERIFIABLE - Insufficient live web evidence available to form a definitive verdict.

 5. Run the Streamlit Application
```bash
streamlit run app/streamlit_app.py
```
