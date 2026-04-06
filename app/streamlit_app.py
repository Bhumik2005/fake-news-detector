import streamlit as st
import requests

# 🔥 IMPORTANT: PUT YOUR DEPLOYED BACKEND URL HERE
API_URL = "https://fake-news-detector-1-lc2z.onrender.com/predict"

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

    # Empty input check
    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")

    else:

        try:
            # 🔎 API CALL
            response = requests.post(
                "https://fake-news-detector-1-lc2z.onrender.com/predict",
                json={"text": text}
            )

            # Debug (optional – remove later)
            # st.write(response.status_code)
            # st.write(response.text)

            if response.status_code == 200:
                data = response.json()

                # Safety check
                if "prediction" not in data:
                    st.error("❌ Invalid response from API")
                    st.write(data)

                else:
                    # 🔥 RESULT
                    st.subheader("Result")

                    if data["prediction"] == "Real":
                        st.success("✅ Verified News")
                    elif data["prediction"] == "Likely Fake":
                        st.error("❌ Likely Fake News")
                    else:
                        st.warning("⚠️ Unverified / Developing Story")

                    # 🔥 MATCH PERCENTAGE
                    st.subheader("Match with Real News")
                    match = data.get("match_percentage", 0)
                    st.progress(int(match))
                    st.caption(f"{match:.1f}% similarity with live news")

                    # 🔥 EXPLANATION
                    st.subheader("Explanation")
                    st.info(data.get("reason", "No explanation available"))

                    # 🔥 RELATED ARTICLES
                    st.subheader("📰 Related News Articles")

                    articles = data.get("related_articles", [])

                    if articles:
                        for article in articles:
                            st.markdown(f"### {article['title']}")
                            st.write(f"Source: {article['source']}")
                            
                            # Published date (if available)
                            if "publishedAt" in article:
                                st.caption(f"Published: {article['publishedAt']}")

                            st.markdown(f"[Read full article]({article['url']})")
                            st.write("---")
                    else:
                        st.write("No relevant news articles found.")

            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.write(response.text)

        except Exception as e:
            st.error("❌ Could not connect to backend")
            st.write(str(e))