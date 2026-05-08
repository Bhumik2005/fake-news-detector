import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000") + "/predict"

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)

# Title
st.title("📰 Real-Time Fake News Detector")
st.write("Enter a news statement or question to verify it using live news sources.")

# Input box
text = st.text_area("Enter News / Question", height=200)

# Predict button
if st.button("Predict"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")

    else:
        if len(text.split()) < 5:
            st.warning("⚠️ Try entering a longer sentence for better accuracy")

        try:
            response = requests.post(API_URL, json={"text": text})

            if response.status_code == 200:
                data = response.json()

                if "prediction" not in data:
                    st.error("❌ Invalid response from API")
                    st.write(data)

                else:
                    st.subheader("Result")

                    if data["prediction"] == "Real":
                        st.success("✅ Verified News")
                    elif data["prediction"] == "Likely Fake":
                        st.error("❌ Likely Fake News")
                    else:
                        st.warning("⚠️ Unverified / Developing Story")

                    st.subheader("Match with Real News")
                    match = data.get("confidence_score", 0)
                    st.progress(int(match))
                    st.caption(f"{match:.1f}% confidence score")

                    ml = data.get("ml_details", {})
                    news = data.get("news_details", {})

                    if ml and news:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Model says", ml.get("label", "—"))
                            st.caption(
                                f"Fake: {ml.get('fake_probability', 0)}% / "
                                f"Real: {ml.get('real_probability', 0)}%"
                            )
                        with col2:
                            st.metric("News similarity", f"{news.get('similarity_score', 0)}%")
                            st.caption(f"Articles found: {news.get('articles_found', 0)}")

                    st.subheader("Explanation")
                    st.info(data.get("reason", "No explanation available"))

                    st.subheader("📰 Related News Articles")
                    articles = data.get("related_articles", [])

                    if articles:
                        for article in articles:
                            st.markdown(f"### {article['title']}")
                            st.write(f"Source: {article['source']}")
                            if "publishedAt" in article:
                                st.caption(f"Published: {article['publishedAt']}")
                            st.markdown(f"[Read full article]({article['url']})")
                            st.write("---")
                    else:
                        st.write("No relevant news articles found.")

                    # 🔍 EXPLAINABILITY
                    st.subheader("🔍 Why did the model decide this?")

                    with st.spinner("Generating explanation..."):
                        explain_response = requests.post(
                            API_URL.replace("/predict", "/explain"),
                            json={"text": text, "use_lime": True}
                        )

                    if explain_response.status_code == 200:
                        exp = explain_response.json()

                        method = exp.get("explanation_method", "unknown")
                        st.caption(f"Explanation method: `{method}`")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**🔴 Words pushing toward Fake**")
                            fake_words = exp.get("top_fake_words", [])
                            if fake_words:
                                for item in fake_words:
                                    weight = item["weight"]
                                    bar = "█" * min(int(weight * 50), 20)
                                    st.markdown(
                                        f"`{item['word']}` &nbsp; "
                                        f"<span style='color:#dc3545;font-size:11px'>"
                                        f"{bar} {weight:.3f}</span>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.caption("No strong fake signals found.")

                        with col2:
                            st.markdown("**🟢 Words pushing toward Real**")
                            real_words = exp.get("top_real_words", [])
                            if real_words:
                                for item in real_words:
                                    weight = item["weight"]
                                    bar = "█" * min(int(weight * 50), 20)
                                    st.markdown(
                                        f"`{item['word']}` &nbsp; "
                                        f"<span style='color:#198754;font-size:11px'>"
                                        f"{bar} {weight:.3f}</span>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.caption("No strong real signals found.")

                        highlighted = exp.get("highlighted_html", "")
                        if highlighted:
                            st.markdown("**Highlighted input** (hover words for scores)")
                            st.markdown(
                                f"<div style='background:#f8f9fa;padding:14px;"
                                f"border-radius:8px;border:1px solid #dee2e6;"
                                f"line-height:2;font-size:15px'>"
                                f"{highlighted}</div>",
                                unsafe_allow_html=True,
                            )

                        top_words = exp.get("top_words", [])
                        if top_words:
                            st.markdown("**Top TF-IDF features in your input**")
                            for item in top_words[:8]:
                                st.markdown(
                                    f"- `{item['word']}` — score: `{item['score']}`"
                                )
                    else:
                        st.caption("Explanation unavailable.")

            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.write(response.text)

        except Exception as e:
            st.error("❌ Could not connect to backend")
            st.write(str(e))