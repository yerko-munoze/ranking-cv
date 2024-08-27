import re
import streamlit as st
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#import pypdf
from pypdf import PdfReader

# Descargar recursos de NLTK
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords') # Eliminar palabras tipicas

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
    'cloud': ['AWS', 'Azure', 'DevOps', 'Terraform', 'IaC', 'SaaS', 'PaaS', 'Kubernetes', 'Virtualización'],
    'recursos humanos': ['RRHH', 'gestión de personal', 'capital humano', 'talento humano', 'personal', 'empleado', 'colaborador', 'trabajador'],
    'reclutamiento' : ['selección', 'contratación', 'búsqueda de talento', 'headhunting', 'onboarding'],
    'liderazgo': ['motivación', 'trabajo en equipo', 'comunicación', 'resolución de conflictos'],
    'desarrollo web': ['CSS','HTML', 'JavaScript', 'front-end', 'desarrollo web', 'maquetación', 'diseño web', 'interfaz de usuario', 'UX', 'UI'],
    'React': ['Angular', 'Vue.js', 'framework', 'biblioteca', 'JavaScript', 'front-end'],
    'Node.js': ['back-end', 'JavaScript', 'servidor', 'API', 'REST', 'Express.js']
}

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return lemmatized_words 

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        if page_num == 0:
            text += page_text
        else:
            text += (page_text + " ") * 2
    return text

def calculate_experience_weight(text):
    experience_keywords = ['años de experiencia', 'años']
    total_years = 0
    for keyword in experience_keywords:
        match = re.search(rf'(\d+)\s*{keyword}', text)
        if match:
            total_years += int(match.group(1))
    return min(total_years / 10, 1)

def calculate_keyword_match(job_description_words, resume_words, essential_keywords):
    # Expandir palabras clave con sinónimos
    expanded_job_keywords = set()
    for word in job_description_words:
        expanded_job_keywords.add(word)
        if word in synonyms:
            expanded_job_keywords.update(synonyms[word])

    common_words = set(expanded_job_keywords) & set(resume_words)
    essential_matches = set(essential_keywords) & common_words

    essential_weight = 2
    match_score = (len(common_words) + essential_weight * len(essential_matches)) / len(job_description_words)
    return match_score

def analyze_experience(job_description, resume_text):

    # Aqui se puede implementar un algoritmo más avanzado

    return 0

# Interfaz de usuario de Streamlit
st.title("CV Match")

# Input descripción del trabajo
job_description = st.text_area("Ingrese una descripción de trabajo", height=100)

# Input de palabras clave esenciales
essential_keywords_input = st.text_input("Ingrese palabras clave esenciales separadas por comas", "Ejemplo: python, sql, power bi, reportes")
essential_keywords = [word.strip() for word in essential_keywords_input.split(",")]

# Input para múltiples PDFs de CVs
uploaded_pdfs = st.file_uploader("Subir CVs (Formato PDF)", type="pdf", accept_multiple_files=True)

if job_description and uploaded_pdfs:
    job_description_processed = preprocess_text(job_description)

    rankings = []

    for uploaded_pdf in uploaded_pdfs:
        resume_text = extract_text_from_pdf(uploaded_pdf)
        resume_processed = preprocess_text(resume_text)

        experience_weight = calculate_experience_weight(resume_text)
        keyword_match = calculate_keyword_match(job_description_processed, resume_processed, essential_keywords)
        experience_match = analyze_experience(job_description, resume_text)

        similarity_adjusted = keyword_match * (1 + 0.1 * experience_weight + 0.2 * experience_match)

        rankings.append({
            "Archivo": uploaded_pdf.name,
            "Puntaje Match": similarity_adjusted * 100
        })

    df_ranking = pd.DataFrame(rankings).sort_values(by="Puntaje Match", ascending=False)

    # Mejorar la visualización de la tabla
    st.dataframe(df_ranking.style.format({'Puntaje Match': '{:.2f}%'}), height=300, use_container_width=True)
else:
    st.warning("Por favor ingrese una descripción y suba al menos un CV en formato PDF")
