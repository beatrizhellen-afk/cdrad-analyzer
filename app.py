import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Título do seu software
st.title("CDRAD Analyzer Pro ⚛️")
st.write("Software de análise automatizada do fantoma CDRAD para otimização de doses.")

# Widget de Upload
arquivo_dicom = st.file_uploader("Faça o upload da imagem DICOM do CDRAD (.dcm)", type=["dcm"])

if arquivo_dicom is not None:
    st.success("Arquivo carregado com sucesso!")
    
    # Lendo o arquivo com pydicom
    dcm = pydicom.dcmread(arquivo_dicom)
    
    # Extraindo metadados ajustados
    st.subheader("Informações Básicas")
    st.write(f"**Equipamento (Fabricante):** {dcm.get('Manufacturer', 'Desconhecido')}")
    
    # Extraindo a matriz de pixels
    imagem_matriz = dcm.pixel_array
    
    # Mostrando a resolução de forma mais amigável
    linhas, colunas = imagem_matriz.shape
    st.write(f"**Resolução da Imagem:** {colunas} x {linhas} pixels")
    
    # --- NOVA FUNCIONALIDADE: MOSTRAR A IMAGEM ---
    st.subheader("Visualização do Fantoma")
    
    # Criando a figura com fundo escuro padrão de raio-x
    fig, ax = plt.subplots(figsize=(8, 8))
    # cmap='gray' garante que a imagem fique em preto e branco como no hospital
    ax.imshow(imagem_matriz, cmap='gray') 
    ax.axis('off') # Remove as bordas e numerações do gráfico para ficar limpo
    
    # Mostrando a imagem no Streamlit
    st.pyplot(fig)
    
    # ---------------------------------------------
    
    st.info("Próximo passo: Iniciar a detecção matemática da grade do CDRAD e cálculo do IQF!")
