import streamlit as st
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Analisis Sentimen Zonasi Sekolah",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 Analisis Sentimen Kebijakan Zonasi Sekolah")
st.markdown("**Proyek Akhir Text Mining – Kelompok 3 PTIK FKIP UNS**")
st.markdown("Alifian Sultan Basundara | Bagus Satyo Nugroho | Ridwan Hakim Mashadi")
st.divider()

@st.cache_resource
def load_models():
    tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    nb    = pickle.load(open("models/model_nb.pkl", "rb"))
    lr    = pickle.load(open("models/model_lr.pkl", "rb"))
    svm   = pickle.load(open("models/model_svm.pkl", "rb"))
    le    = pickle.load(open("models/label_encoder.pkl", "rb"))
    tok   = AutoTokenizer.from_pretrained("models/indobertweet_tokenizer")
    mdl   = AutoModelForSequenceClassification.from_pretrained("models/indobertweet_model")
    mdl.eval()
    return tfidf, nb, lr, svm, le, tok, mdl

tfidf, nb_m, lr_m, svm_m, le, bert_tok, bert_model = load_models()

LABEL_ICON = {"Positif": "🟢", "Negatif": "🔴", "Netral": "🔵"}

def predict_classical(text, model):
    vec  = tfidf.transform([text])
    pred = model.predict(vec)[0]
    return le.inverse_transform([pred])[0]

def predict_bert(text):
    inputs = bert_tok(text, return_tensors="pt", truncation=True,
                      padding="max_length", max_length=128)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred  = int(np.argmax(probs))
    return le.inverse_transform([pred])[0], probs

col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area(
        "Masukkan teks tweet tentang zonasi sekolah:",
        height=120,
        placeholder="Contoh: Kebijakan zonasi sekolah sangat tidak adil..."
    )
with col2:
    model_choice = st.selectbox("Pilih Model:", [
        "IndoBERTweet (Transformer)",
        "Naive Bayes (TF-IDF)",
        "Logistic Regression (TF-IDF)",
        "SVM – Linear (TF-IDF)",
        "Semua Model"
    ])

if st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        st.subheader("Hasil Prediksi")
        if model_choice == "Semua Model":
            cols = st.columns(4)
            for col, (mname, pred) in zip(cols, [
                ("Naive Bayes",       predict_classical(user_input, nb_m)),
                ("Log. Regression",   predict_classical(user_input, lr_m)),
                ("SVM",               predict_classical(user_input, svm_m)),
                ("IndoBERTweet",      predict_bert(user_input)[0]),
            ]):
                col.metric(mname, f"{LABEL_ICON.get(pred, '')} {pred}")
        elif "IndoBERTweet" in model_choice:
            label, probs = predict_bert(user_input)
            st.metric("IndoBERTweet", f"{LABEL_ICON.get(label, '')} {label}")
            st.bar_chart(dict(zip(le.classes_, probs.tolist())))
        elif "Naive Bayes" in model_choice:
            label = predict_classical(user_input, nb_m)
            st.metric("Naive Bayes", f"{LABEL_ICON.get(label, '')} {label}")
        elif "Logistic" in model_choice:
            label = predict_classical(user_input, lr_m)
            st.metric("Logistic Regression", f"{LABEL_ICON.get(label, '')} {label}")
        elif "SVM" in model_choice:
            label = predict_classical(user_input, svm_m)
            st.metric("SVM", f"{LABEL_ICON.get(label, '')} {label}")

st.divider()
st.caption("PTIK FKIP UNS – Text Mining 2026 | Dosen: Yudianto Sujana, S.Kom., M.Kom.")
