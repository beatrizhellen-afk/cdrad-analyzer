import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="CDRAD Analyzer Pro", layout="wide")

st.title("CDRAD Analyzer Pro ⚛️")
st.write("Software de análise automatizada do fantoma CDRAD para otimização de doses.")

# Valores padrão do CDRAD (Diâmetros e Profundidades em mm)
dimensoes = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 8.0]

# --- MENU LATERAL DE ORIENTAÇÃO ---
st.sidebar.subheader("Correção de Orientação")
st.sidebar.write("Use os controles abaixo se o DICOM estiver invertido.")
espelhar_h = st.sidebar.checkbox("↔️ Espelhar Horizontalmente")
espelhar_v = st.sidebar.checkbox("↕️ Espelhar Verticalmente")
rotacionar = st.sidebar.checkbox("🔄 Rotacionar 90º")

arquivo_dicom = st.file_uploader("Faça o upload da imagem DICOM do CDRAD (.dcm)", type=["dcm"])

if arquivo_dicom is not None:
    dcm = pydicom.dcmread(arquivo_dicom)
    imagem_bruta = dcm.pixel_array
    
    # --- APLICANDO A CORREÇÃO DE ORIENTAÇÃO ---
    if espelhar_h:
        imagem_bruta = np.fliplr(imagem_bruta) # Inverte Esquerda/Direita
    if espelhar_v:
        imagem_bruta = np.flipud(imagem_bruta) # Inverte Cima/Baixo
    if rotacionar:
        imagem_bruta = np.rot90(imagem_bruta)  # Gira 90 graus
    
    # Normalização e Realce
    imagem_norm = cv2.normalize(imagem_bruta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imagem_realcada = clahe.apply(imagem_norm)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Análise da Grade CDRAD")
        
        h, w = imagem_realcada.shape
        cell_h, cell_w = h // 15, w // 15
        
        contagem_vistos = 0
        matriz_deteccao = np.zeros((15, 15))
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(imagem_realcada, cmap='gray')

        for i in range(15): 
            for j in range(15): 
                y, x = i * cell_h, j * cell_w
                roi = imagem_realcada[y:y+cell_h, x:x+cell_w]
                
                # Calibração de Sensibilidade
                if np.std(roi) > 12: 
                    matriz_deteccao[i, j] = 1
                    contagem_vistos += 1
                    rect = plt.Rectangle((x, y), cell_w, cell_h, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
                else:
                    rect = plt.Rectangle((x, y), cell_w, cell_h, linewidth=1, edgecolor='r', facecolor='none', alpha=0.2)
                
                ax.add_patch(rect)
        
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        st.subheader("Resultados da Otimização")
        
        st.metric("Total de Quadrados Detectados", f"{contagem_vistos} / 225")
        
        indices_detectados = np.where(matriz_deteccao == 1)
        if len(indices_detectados[1]) > 0:
            menor_dia = dimensoes[np.min(indices_detectados[1])]
            menor_prof = dimensoes[np.min(indices_detectados[0])]
        else:
            menor_dia, menor_prof = 0, 0

        st.write(f"**Resolução de Detalhe (Menor Furo):** {menor_dia} mm")
        st.write(f"**Threshold de Contraste (Mais Raso):** {menor_prof} mm")

        iqf = 0
        for j in range(15):
            detectados_na_coluna = np.where(matriz_deteccao[:, j] == 1)[0]
            if len(detectados_na_coluna) > 0:
                prof_limite = dimensoes[np.min(detectados_na_coluna)]
                iqf += dimensoes[j] * prof_limite
        
        st.divider()
        st.subheader("Índice de Qualidade")
        st.title(f"IQF: {iqf:.2f}")
