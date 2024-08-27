import re
import streamlit as st
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import pypdf
from unidecode import unidecode

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Inicializar lematizador y stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

# Diccionario de sinónimos
synonyms = {
    'informática': ['tecnología', 'computación', 'sistemas'],
    'python': ['programación', 'software'],
    'sql': ['base de datos', 'consultas', 'bases de datos relacionales'],
    'power bi': ['visualización', 'análisis de datos', 'dashboard', 'informes'],
    'desarrollo': ['creación', 'programación', 'implementación'],
    'reportes': ['informes', 'análisis'],
    'cloud': ['AWS', 'Azure'],
    'comercial': ['recursos humanos', 'economía', 'administración', 'gestión']
}

def preprocess_text(text):
    text = unidecode(text)  # Eliminar acentos
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(lemmatized_words)

def extract_text_from_pdf(pdf_file):
    pdf_reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        text += page_text
    return text

def calculate_cosine_similarity(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_similarities, tfidf_matrix

def apply_knn(tfidf_matrix, n_neighbors=5):
    n_samples = tfidf_matrix.shape[0]
    n_neighbors = min(n_neighbors + 1, n_samples)  # +1 para incluir el propio documento

    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn_model.fit(tfidf_matrix)
    distances, indices = knn_model.kneighbors(tfidf_matrix)
    return distances[:, 1:], indices[:, 1:]  # Omitir la primera columna que corresponde al propio documento

def rank_with_knn(cosine_similarities, indices):
    knn_scores = []
    for i, neighbors_indices in enumerate(indices):
        valid_indices = [idx-1 for idx in neighbors_indices if idx-1 >= 0 and idx-1 < len(cosine_similarities)]
        if valid_indices:
            knn_scores.append(cosine_similarities[valid_indices].mean())
        else:
            knn_scores.append(0)
    return knn_scores

# Interfaz de usuario de Streamlit
st.title("CV Match")

job_description = st.text_area("Ingrese una descripción de trabajo", height=100)
uploaded_pdfs = st.file_uploader("Subir CVs (Formato PDF)", type="pdf", accept_multiple_files=True)

if job_description and uploaded_pdfs:
    job_description_processed = preprocess_text(job_description)
    resumes_texts = [preprocess_text(extract_text_from_pdf(pdf)) for pdf in uploaded_pdfs]

    cosine_similarities, tfidf_matrix = calculate_cosine_similarity(job_description_processed, resumes_texts)
    distances, indices = apply_knn(tfidf_matrix)

    knn_scores = rank_with_knn(cosine_similarities, indices)

    rankings = []
    for i, uploaded_pdf in enumerate(uploaded_pdfs):
        rankings.append({
            "Archivo": uploaded_pdf.name,
            "Puntaje Match": knn_scores[i] * 100
        })

    df_ranking = pd.DataFrame(rankings).sort_values(by="Puntaje Match", ascending=False)
    st.dataframe(df_ranking.style.format({'Puntaje Match': '{:.2f}%'}), height=300, use_container_width=True)

else:
    st.warning("Por favor ingrese una descripción y suba al menos un CV en formato PDF")



