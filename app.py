import json
import re

import nltk
import streamlit as st
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2563EB;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #6B7280;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

stemmer = SnowballStemmer("english")


def preprocess_text(text):
    """Lowercase, tokenize, remove punctuation/stopwords, and stem words."""
    text = text.lower()
    tokens = wordpunct_tokenize(text)

    cleaned_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token.isalpha() and token not in ENGLISH_STOP_WORDS
    ]

    return " ".join(cleaned_tokens)


@st.cache_data
def load_faqs():
    with open("faqs.json", "r", encoding="utf-8") as file:
        return json.load(file)


def find_best_answer(user_question, faqs, vectorizer, faq_vectors):
    user_vector = vectorizer.transform([user_question])

    similarity_scores = cosine_similarity(user_vector, faq_vectors).flatten()

    best_index = similarity_scores.argmax()
    confidence_score = similarity_scores[best_index]

    # If no FAQ question is similar enough
    if confidence_score < 0.20:
        return (
            "Sorry, I could not find an answer for that question. "
            "Please try asking in a different way.",
            confidence_score
        )

    return faqs[best_index]["answer"], confidence_score


faqs = load_faqs()
faq_questions = [faq["question"] for faq in faqs]

# Convert FAQ questions into TF-IDF vectors
vectorizer = TfidfVectorizer(
    preprocessor=preprocess_text,
    ngram_range=(1, 2)
)
faq_vectors = vectorizer.fit_transform(faq_questions)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("About this project")
    st.write(
        "This chatbot finds the best FAQ answer using "
        "NLP preprocessing, TF-IDF, and cosine similarity."
    )

    st.subheader("Try asking")
    st.write("- How do I reset my password?")
    st.write("- Can I cancel my order?")
    st.write("- How can I track my order?")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown("<h1 class='main-title'>🤖 FAQ Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Ask a question about our online store.</p>",
    unsafe_allow_html=True
)

# Show old chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Type your question here...")

if user_question:
    # Display and save user message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    # Find and display chatbot answer
    answer, confidence = find_best_answer(
        user_question,
        faqs,
        vectorizer,
        faq_vectors
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)

        if confidence >= 0.20:
            st.caption(f"Match confidence: {confidence * 100:.0f}%")