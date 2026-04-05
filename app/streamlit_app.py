import streamlit as st
import requests

# 🔥 IMPORTANT: USE YOUR DEPLOYED API URL HERE
API_URL = "https://fake-news-detector-1-lc2z.onrender.com/predict"

# Page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Title
st.title("📰 Fake News Detector")
st.write("Enter a news article or question to verify its authenticity.")

# Input
text = st.text_area("Enter News Article", height=200)

# Button
if st.button("Predict"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")

    else:
        if len(text.split()) < 5:
            st.warning("⚠️ Try entering a longer sentence for better results")

        try:
            response = requests.post(
                API_URL,
                json={"text": text}
            )

            if response.status_code == 200:
                data = response.json()

                if "error" in data:
                    st.error(data["error"])
                else:
                    # 🔥 RESULT
                    st.subheader("Result")

                    if data["prediction"] == "Real":
                        st.success("✅ Real News")
                    elif data["prediction"] == "Fake":
                        st.error("❌ Fake News")
                    else:
                        st.warning("⚠️ Uncertain")

                    # 🔥 REASON
                    st.subheader("Why this result?")
                    st.info(data["reason"])

                    # 🔥 CONFIDENCE
                    st.subheader("Model Confidence")

                    st.write("Real News Confidence")
                    st.progress(int(data["confidence_real"] * 100))
                    st.caption(f"{data['confidence_real']*100:.1f}%")

                    st.write("Fake News Confidence")
                    st.progress(int(data["confidence_fake"] * 100))
                    st.caption(f"{data['confidence_fake']*100:.1f}%")

                    # 🔥 SIMILARITY
                    st.subheader("Similarity with live news")
                    st.progress(int(data["similarity_score"] * 100))
                    st.caption(f"{data['similarity_score']*100:.1f}% match")

                    # 🔥 RELATED ARTICLES
                    st.subheader("📰 Related News Articles")

                    articles = data.get("related_articles", [])

                    if articles:
                        for article in articles:
                            st.markdown(f"**{article['title']}**")
                            st.write(f"Source: {article['source']}")
                            st.markdown(f"[Read more]({article['url']})")
                            st.write("---")
                    else:
                        st.write("No related news found.")

            else:
                st.error("❌ API Error")

        except Exception as e:
            st.error("❌ Could not connect to backend")
            st.write(str(e))