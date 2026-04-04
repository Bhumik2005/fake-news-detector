import streamlit as st
import requests

# Page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# API URL (DEPLOYED BACKEND)
API_URL = "https://fake-news-detector-1-lc2z.onrender.com/predict"

# Title
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's real or fake.")

# Input box
text = st.text_area("Enter News Article", height=200)

# Button
if st.button("Predict"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")

    else:
        if len(text.split()) < 20:
            st.warning("⚠️ Please enter a longer article for better accuracy")

        try:
            response = requests.post(API_URL, json={"text": text})

            if response.status_code == 200:
                data = response.json()

                st.subheader("Result")
                if data["prediction"] == 1:
                    st.success("✅ Real News")
                else:
                    st.error("❌ Fake News")

                st.subheader("Prediction Confidence")

                st.write("Real News Confidence")
                st.progress(int(data["confidence_real"] * 100))
                st.caption(f"{data['confidence_real']*100:.1f}%")

                st.write("Fake News Confidence")
                st.progress(int(data["confidence_fake"] * 100))
                st.caption(f"{data['confidence_fake']*100:.1f}%")

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
            st.error("❌ Could not connect to backend.")
            st.write(str(e))