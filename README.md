# 🚀 AI Resume Auditor & Semantic Ranker

This project is an advanced **AI Resume Screening Tool** designed to bridge the gap between unstructured resumes and technical job descriptions. Unlike standard keyword matchers, this system uses **Deep Learning** to understand the "context" of a candidate's experience—recognizing, for example, that your work with **Supabase** qualifies as **NoSQL Database** experience.

## 📖 What the Code Does

The application follows a four-stage intelligent pipeline to assess candidate relevance:

### 1. PDF Text Extraction
The system utilizes **PyMuPDF** to parse raw text from uploaded PDF resumes. It handles various layouts, ensuring that technical details—like your **9.12 CGPA** or **AWS Academy** certifications—are captured accurately.

### 2. Natural Language Preprocessing (NLP)
Using **SpaCy**, the code cleans the text by:
**Lemmatization:** Converting words to their root form (e.g., "databases" becomes "database") to ensure exact matches aren't missed.
**Noise Filtering:** Stripping out "filler" words like "responsibilities," "requirements," or "background" to focus strictly on engineering skills.

### 3. Semantic Vectorization (The "Brain")
This is the core of the project. Instead of just looking for words, it uses a **Sentence Transformer model** (`paraphrase-multilingual-MiniLM-L12-v2`) to convert text into high-dimensional vectors. 
* It calculates **Cosine Similarity** to see how closely the "vibe" of your resume matches the job's needs.

### 4. Smart Skill Mapping
The code includes a custom **Synonym Engine**. If a Job Description asks for "Generative AI" and your resume lists **"Gemini API"**, the system automatically recognizes this as a match.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Web UI)
* **AI Models:** Sentence-Transformers (SBERT)
* **NLP:** SpaCy
* **PDF Engine:** PyMuPDF
* **Data Science:** Pandas & Scikit-Learn

## 📊 Summary of Results
The auditor provides:
* **Match Score:** A hybrid percentage based on semantic context (70%) and hard keyword matches (30%).
* **Skill Audit:** A list of technical terms found in the resume.
* **Gap Analysis:** Specific suggestions for keywords or skills to add to improve the resume's visibility to ATS (Applicant Tracking Systems).
