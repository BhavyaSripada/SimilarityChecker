import streamlit as st
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

st.title("Sentence Similarity Checker")

# Example sentences
sentence1 = st.text_input("Enter sentence 1:")
sentence2 = st.text_input("Enter sentence 2:")

# Compute embeddings for the sentences
embeddings1 = model.encode(sentence1, convert_to_tensor=True)
embeddings2 = model.encode(sentence2, convert_to_tensor=True)

# Calculate cosine similarity
cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()

if cosine_score > 0.8:
    st.markdown('<span style="color:green">&#10004;</span>', unsafe_allow_html=True)
else:
    st.markdown('<span style="color:red">&#10008;</span>', unsafe_allow_html=True)

st.write(f"Similarity score: {cosine_score:.4f}")
