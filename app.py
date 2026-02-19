import streamlit as st
import nltk
import math
import re
import speech_recognition as sr

# If you didn't download punkt yet, uncomment the next line once:
# nltk.download("punkt")

# -----------------------------
# 1) Load and preprocess text file
# -----------------------------
def load_corpus(file_path="data.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    # split into sentences
    sentences = nltk.sent_tokenize(text)
    return sentences

def tokenize(text):
    # lowercase + keep only letters/numbers/spaces
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    return tokens

# Build IDF values for all sentences (TF-IDF, but implemented simply)
def build_idf(sentences):
    N = len(sentences)
    df = {}  # document frequency
    tokenized_sentences = []

    for s in sentences:
        tokens = tokenize(s)
        tokenized_sentences.append(tokens)
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df[t] = df.get(t, 0) + 1

    idf = {}
    for t, freq in df.items():
        # smooth IDF
        idf[t] = math.log((N + 1) / (freq + 1)) + 1

    return tokenized_sentences, idf

def tfidf_vector(tokens, idf):
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    vec = {}
    length = max(len(tokens), 1)
    for t, count in tf.items():
        vec[t] = (count / length) * idf.get(t, 0.0)

    return vec

def cosine_similarity(vec1, vec2):
    # dot product
    dot = 0.0
    for t, v in vec1.items():
        dot += v * vec2.get(t, 0.0)

    # norms
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def chatbot_response(user_text, sentences, tokenized_sentences, idf):
    user_tokens = tokenize(user_text)
    user_vec = tfidf_vector(user_tokens, idf)

    best_score = 0.0
    best_idx = -1

    for i, sent_tokens in enumerate(tokenized_sentences):
        sent_vec = tfidf_vector(sent_tokens, idf)
        score = cosine_similarity(user_vec, sent_vec)
        if score > best_score:
            best_score = score
            best_idx = i

    # If the best match is too weak, give a fallback answer
    if best_score < 0.15:
        return "I’m not sure I understand. Can you rephrase your question?"

    return sentences[best_idx]

# -----------------------------
# 2) Speech to text function
# -----------------------------
def transcribe_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, timeout=5, phrase_time_limit=8)

    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return "__API_ERROR__"

# -----------------------------
# 3) Streamlit App UI
# -----------------------------
st.title("🎙️ Speech-Enabled Chatbot (Simple)")

# Load chatbot data once
@st.cache_data
def init_chatbot():
    sentences = load_corpus("data.txt")
    tokenized_sentences, idf = build_idf(sentences)
    return sentences, tokenized_sentences, idf

sentences, tokenized_sentences, idf = init_chatbot()

st.write("Choose input type:")

mode = st.radio("Input mode", ["Text", "Speech"], horizontal=True)

user_text = ""

if mode == "Text":
    user_text = st.text_input("Type your message:")
    if st.button("Send"):
        if user_text.strip():
            bot = chatbot_response(user_text, sentences, tokenized_sentences, idf)
            st.markdown(f"**You:** {user_text}")
            st.markdown(f"**Bot:** {bot}")

else:
    st.write("Click the button and speak into your microphone.")
    if st.button("🎤 Record"):
        result = transcribe_speech()

        if result is None:
            st.error("Sorry, I couldn't understand your speech. Try again.")
        elif result == "__API_ERROR__":
            st.error("Speech recognition API error (internet issue or service unavailable).")
        else:
            user_text = result
            bot = chatbot_response(user_text, sentences, tokenized_sentences, idf)
            st.markdown(f"**You (transcribed):** {user_text}")
            st.markdown(f"**Bot:** {bot}")

st.caption("Note: Voice input works when you run Streamlit locally on the same computer that has the microphone.")
