import streamlit as st

from sentiment_model import SentimentAnalyzer


st.set_page_config(
    page_title="RoBERTa Sentiment Analysis",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource
def load_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


st.title("RoBERTa Sentiment Analysis")

st.write(
    "Enter some text to classify its sentiment as "
    "negative, neutral or positive."
)

user_text = st.text_area(
    label="Text to analyze",
    placeholder="Example: I really enjoyed using this product!",
    height=150,
)

analyze_button = st.button(
    "Analyze sentiment",
    type="primary",
    use_container_width=True,
)

if analyze_button:
    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            with st.spinner("Analyzing sentiment..."):
                analyzer = load_analyzer()
                result = analyzer.predict(user_text)

            st.success(f"Predicted sentiment: {result['label']}")

            prediction_column, confidence_column = st.columns(2)

            prediction_column.metric(
                label="Sentiment",
                value=result["label"],
            )

            confidence_column.metric(
                label="Confidence",
                value=f"{result['confidence']:.2%}",
            )

            st.subheader("Class probabilities")

            for label, score in result["scores"].items():
                st.write(f"{label}: {score:.2%}")
                st.progress(score)

            st.caption(f"Inference device: {result['device'].upper()}")

        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Analysis failed: {error}")