import os
import torch
import torch.nn as nn
import streamlit as st
import fitz
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    # Using L12 for deeper semantic understanding
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return nlp, sbert_model

nlp, sbert_model = load_models()

# --- 2. ENHANCED SKILL MAPPING ---
SKILL_SYNONYMS = {
    "supabase": ["database", "nosql", "postgresql", "backend", "storage", "databases"],
    "gemini": ["ai", "llm", "generative", "intelligence", "nlp", "llms", "models"],
    "fastapi": ["api", "backend", "rest", "python", "apis"],
    "react": ["frontend", "javascript", "ui", "web"],
    "lambda": ["serverless", "cloud", "aws", "functions"],
    "s3": ["storage", "cloud", "aws", "s3", "buckets"],
    "computer engineering": ["academics", "engineering", "student", "background"]
}

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_expanded_keywords(text):
    """Extracts keywords and normalizes plurals to bridge the gap."""
    doc = nlp(text.lower())
    
    # 1. Extract and Lemmatize (e.g., 'databases' -> 'database')
    keywords = set([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop])
    
    # 2. Hard-filter 'Filler' words that aren't technical skills
    custom_ignore = {
        'responsibilities', 'requirements', 'familiarity', 'background', 
        'overview', 'title', 'junior', 'tasks', 'experiences', 'workflow',
        'interfaces', 'platforms', 'developer', 'functions', 'user', 
        'candidate', 'role', 'associate', 'academics', 'proficient', 'stack',
        'associate', 'proficient', 'optimization', 'workflows'
    }
    
    clean_keywords = {word for word in keywords if word not in custom_ignore}

    # 3. Apply Synonym Mapping
    expanded = set(clean_keywords)
    for word in clean_keywords:
        if word in SKILL_SYNONYMS:
            expanded.update(SKILL_SYNONYMS[word])
            
    return expanded

def get_detailed_analysis(resume_text, jd_text):
    # Semantic Match (Weighted for technical depth)
    resume_emb = sbert_model.encode(resume_text, convert_to_tensor=True)
    jd_emb = sbert_model.encode(jd_text, convert_to_tensor=True)
    semantic_score = float(util.pytorch_cos_sim(resume_emb, jd_emb).item() * 100)
    
    # Keyword Match
    resume_ks = get_expanded_keywords(resume_text)
    jd_ks = get_expanded_keywords(jd_text)
    
    matched = resume_ks.intersection(jd_ks)
    missing = jd_ks - resume_ks
    
    # Final score is a mix of Semantic 'Vibe' and Hard Keyword Match
    final_score = (semantic_score * 0.7) + (len(matched) / max(len(jd_ks), 1) * 30)
    
    return min(100, final_score), list(matched), list(missing)

# --- 3. UI SETUP ---
st.set_page_config(page_title="Resume Optimizer Pro", layout="wide")
st.title("🧠 Advanced Resume Optimizer")

jd_input = st.text_area("Paste Job Description", height=200)
uploaded_files = st.file_uploader("Upload Resume PDF", type="pdf", accept_multiple_files=True)

if st.button("Run Advanced Audit"):
    if jd_input and uploaded_files:
        for file in uploaded_files:
            raw_text = extract_text_from_pdf(file)
            score, matched, missing = get_detailed_analysis(raw_text, jd_input)
            
            st.header(f"Analysis for: {file.name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Match Score", f"{score:.2f}%")
            c2.success(f"✅ {len(matched)} Skills Matched")
            c3.error(f"❌ {len(missing)} Potential Gaps")

            with st.expander("Why this score?"):
                t1, t2 = st.tabs(["Matches Found", "Critical Missing Terms"])
                with t1:
                    st.write("Recognized Skills & Synonyms:")
                    st.caption(", ".join(matched))
                with t2:
                    st.write("Keywords that didn't have an exact match:")
                    st.caption(", ".join(list(missing)[:20]))
            st.divider()