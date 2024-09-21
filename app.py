# Graphical Interface
import streamlit as st
import pandas as pd
from transformation import *

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
