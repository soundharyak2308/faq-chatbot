# FAQ Chatbot

A Streamlit-based FAQ chatbot for an online store. It finds the best answer to a user's question using Natural Language Processing.

## Features

- Interactive chat interface
- FAQ data stored in JSON format
- Text preprocessing using NLTK
- TF-IDF vectorization
- Cosine similarity for FAQ matching
- Match confidence score
- Fallback answer for unrelated questions
- Clear chat button

## Technologies Used

- Python
- Streamlit
- NLTK
- Scikit-learn
- JSON

## How It Works

1. The chatbot loads FAQ questions and answers from `faqs.json`.
2. It preprocesses each question by lowercasing, tokenizing, removing stop words, and stemming.
3. TF-IDF converts questions into numeric vectors.
4. Cosine similarity compares the user’s question with stored FAQ questions.
5. The chatbot displays the answer with the highest similarity score.

## Run the Project

Install dependencies:

```bash
python -m pip install -r requirements.txt