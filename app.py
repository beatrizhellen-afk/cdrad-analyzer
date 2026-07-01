import streamlit as st
import pydicom
import numpy as np

# Título do seu software
st.title("CDRAD Analyzer Pro ⚛️")
st.write("Software de análise automatizada do fantoma CDRAD para otimização de doses.")

# Widget de Upload
arquivo_dicom = st.file_uploader("Faça o upload da imagem DICOM do CDRAD (.dcm)", type=["dcm"])

if arquivo_dicom is not None:
    st.success("Arquivo carregado com sucesso!")
    
    # Lendo o arquivo com pydicom
    dcm = pydicom.dcmread(arquivo_dicom)
    
    # Extraindo algumas informações para mostrar na tela
    st.subheader("Metadados da Imagem")
    st.write(f"**Equipamento:** {dcm.get('Manufacturer', 'Desconhecido')}")
    st.write(f"**kVp utilizado:** {dcm.get('KVP', 'Não informado')} kV")
    st.write(f"**mAs utilizado:** {dcm.get('Exposure', 'Não informado')} mAs")
    
    # Extraindo a matriz de pixels (A imagem bruta)
    imagem_matriz = dcm.pixel_array
    
    # Mostrando a resolução
    st.write(f"**Resolução da Matriz:** {imagem_matriz.shape}")
    
    st.info("Aqui entrará o seu algoritmo de detecção da grade e cálculo do IQF!")