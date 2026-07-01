import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Título do seu software
st.title("CDRAD Analyzer Pro ⚛️")
st.write("Software de análise automatizada do fantoma CDRAD para otimização de doses.")

# Widget de Upload
arquivo_dicom = st.file_uploader("Faça o upload da imagem DICOM do CDRAD (.dcm)", type=["dcm"])

if arquivo_dicom is not None:
    st.success("Arquivo carregado com sucesso!")
    
    # Lendo o arquivo com pydicom
    dcm = pydicom.dcmread(arquivo_dicom)
    
    st.subheader("Informações Básicas")
    st.write(f"**Equipamento (Fabricante):** {dcm.get('Manufacturer', 'Desconhecido')}")
    
    # Extraindo a matriz de pixels e convertendo para o formato correto do OpenCV
    imagem_bruta = dcm.pixel_array
    linhas, colunas = imagem_bruta.shape
    st.write(f"**Resolução da Imagem:** {colunas} x {linhas} pixels")
    
    # Normalizando a imagem para 8 bits (0-255) para os filtros funcionarem
    imagem_norm = cv2.normalize(imagem_bruta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    st.markdown("---")
    st.subheader("Processamento de Imagem (Visão Computacional)")
    
    # Criando um menu para escolher o filtro
    filtro = st.radio(
        "Escolha o modo de visualização:",
        ("Imagem Original", "Realce de Contraste Máximo (CLAHE)", "Detecção de Bordas (Canny)")
    )
    
    # Aplicando o filtro escolhido
    if filtro == "Imagem Original":
        imagem_processada = imagem_norm
        
    elif filtro == "Realce de Contraste Máximo (CLAHE)":
        # O CLAHE equaliza o histograma em pequenos blocos, revelando furos ocultos
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        imagem_processada = clahe.apply(imagem_norm)
        
    elif filtro == "Detecção de Bordas (Canny)":
        # O Canny desenha as linhas de contorno (ótimo para ver a grade do CDRAD)
        # Primeiro damos um leve desfoque para reduzir o ruído
        suave = cv2.GaussianBlur(imagem_norm, (5, 5), 0)
        imagem_processada = cv2.Canny(suave, 30, 100)
    
    # Renderizando a imagem na tela
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(imagem_processada, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)
    
    st.info("Próximo passo: Segmentar a imagem e extrair a matriz 15x15 do CDRAD!")
