import re
import streamlit as st
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pypdf
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