import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from streamlit_cropper import st_cropper

st.set_page_config(page_title="CDRAD Analyzer Pro", layout="wide")

st.title("CDRAD Analyzer Pro ⚛️")
st.write("Software de análise automatizada do fantoma CDRAD para otimização de doses.")

# Definição EXATA das dimensões do CDRAD 2.0
diametros_y = [8.0, 6.3, 5.0, 4.0, 3.2, 2.5, 2.0, 1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3] # Linhas (Cima -> Baixo)
profundidades_x = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 8.0] # Colunas (Esq -> Dir)

# --- MENU LATERAL ---
st.sidebar.header("🔧 Controles da Imagem")
st.sidebar.subheader("1. Correção de Orientação")
espelhar_h = st.sidebar.checkbox("↔️ Espelhar Horizontalmente")
espelhar_v = st.sidebar.checkbox("↕️ Espelhar Verticalmente")
angulo_rotacao = st.sidebar.number_input("🔄 Alinhamento Fino (Graus)", min_value=-180.0, max_value=180.0, value=0.0, step=0.1, format="%.1f")

st.sidebar.subheader("2. Perfil do Equipamento")
modo_telecomandado = st.sidebar.checkbox("☢️ Modo Telecomandado / Fluoroscopia")
st.sidebar.caption("Ative se a imagem tiver bordas hexagonais ou distorção circular para ignorar zonas mortas.")

st.sidebar.subheader("3. Sensibilidade")
sensibilidade = st.sidebar.slider("Limiar de Detecção (Ruído)", 5, 40, 18) # Aumentado ligeiramente o padrão para maior robustez

arquivo_dicom = st.file_uploader("Faça o upload da imagem DICOM do CDRAD (.dcm)", type=["dcm"])

if arquivo_dicom is not None:
    dcm = pydicom.dcmread(arquivo_dicom)
    imagem_bruta = dcm.pixel_array
    
    if espelhar_h: imagem_bruta = np.fliplr(imagem_bruta)
    if espelhar_v: imagem_bruta = np.flipud(imagem_bruta)
    
    imagem_norm = cv2.normalize(imagem_bruta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if angulo_rotacao != 0.0:
        h_orig, w_orig = imagem_norm.shape
        M = cv2.getRotationMatrix2D((w_orig // 2, h_orig // 2), angulo_rotacao, 1.0)
        imagem_processada = cv2.warpAffine(imagem_norm, M, (w_orig, h_orig), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        imagem_processada = imagem_norm.copy()

    st.markdown("---")
    st.subheader("✂️ Recorte Interativo da Matriz (ROI)")
    
    img_pil = Image.fromarray(imagem_processada)
    imagem_recortada_pil = st_cropper(img_pil, realtime_update=True, box_color='#0000FF', aspect_ratio=(1, 1))
    
    imagem_roi_bruta = np.array(imagem_recortada_pil)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imagem_roi = clahe.apply(imagem_roi_bruta)

    st.markdown("---")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Análise da Grade CDRAD")
        h, w = imagem_roi.shape
        cell_h, cell_w = h // 15, w // 15
        contagem_vistos = 0
        matriz_deteccao = np.zeros((15, 15))
        
        fig_img, ax_img = plt.subplots(figsize=(10, 10))
        ax_img.imshow(imagem_roi, cmap='gray')
        
        for i in range(15): 
            for j in range(15): 
                y, x = i * cell_h, j * cell_w
                
                margem_y = int(cell_h * 0.2)
                margem_x = int(cell_w * 0.2)
                roi_celula = imagem_roi[y+margem_y : y+cell_h-margem_y, x+margem_x : x+cell_w-margem_x]
                
                if roi_celula.size > 0:
                    std_cel = np.std(roi_celula)
                    media_cel = np.mean(roi_celula)
                    
                    detectado = False
                    ignorado = False
                    
                    if modo_telecomandado:
                        # Filtro estrito para artefatos de colimação hexagonal nas bordas
                        if std_cel > 55 or media_cel < 35 or media_cel > 225:
                            ignorado = True
                        elif std_cel > sensibilidade:
                            detectado = True
                    else:
                        if std_cel > sensibilidade:
                            detectado = True

                    if ignorado:
                        ax_img.add_patch(plt.Rectangle((x, y), cell_w, cell_h, linewidth=1, edgecolor='b', facecolor='none', alpha=0.3))
                        ax_img.plot([x, x+cell_w], [y, y+cell_h], color='b', alpha=0.5, linewidth=1)
                        ax_img.plot([x+cell_w, x], [y, y+cell_h], color='b', alpha=0.5, linewidth=1)
                    elif detectado:
                        matriz_deteccao[i, j] = 1
                        contagem_vistos += 1
                        ax_img.add_patch(plt.Rectangle((x, y), cell_w, cell_h, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5))
                    else:
                        ax_img.add_patch(plt.Rectangle((x, y), cell_w, cell_h, linewidth=1, edgecolor='r', facecolor='none', alpha=0.2))
                        
        ax_img.axis('off')
        st.pyplot(fig_img)

    with col2:
        st.subheader("📊 Métricas e Curva C-D")
        
        # --- NOVA LÓGICA FILTRADA E ROBUSTA DE MÉTRICAS ---
        menor_dia = 0.0
        menor_prof = 0.0
        
        # 1. Resolução Espacial (Diâmetro): Encontra a última linha (de cima para baixo) que possui validação consistente
        # Uma linha válida deve ter furos detectados e não pode ser apenas um ruído isolado nas bordas pretas
        linhas_validas = []
        for i in range(15):
            if np.sum(matriz_deteccao[i, :]) >= 2: # Exige pelo menos 2 furos detectados na mesma linha para validá-la
                linhas_validas.append(i)
        if len(linhas_validas) > 0:
            menor_dia = diametros_y[np.max(linhas_validas)]
            
        # 2. Baixo Contraste (Profundidade): Encontra a coluna mais à esquerda com detecção consistente
        colunas_validas = []
        for j in range(15):
            if np.sum(matriz_deteccao[:, j]) >= 2: # Exige pelo menos 2 furos detectados na mesma coluna
                colunas_validas.append(j)
        if len(colunas_validas) > 0:
            menor_prof = profundidades_x[np.min(colunas_validas)]

        met_res_esp, met_threshold = st.columns(2)
        with met_res_esp:
            st.metric("Res. Espacial (Diâmetro)", f"{menor_dia} mm" if menor_dia > 0 else "Não Detectado")
        with met_threshold:
            st.metric("Baixo Contraste (Prof.)", f"{menor_prof} mm" if menor_prof > 0 else "Não Detectado")
        
        st.divider()
        
        # --- CÁLCULO DO IQF E CURVA C-D ---
        iqf = 0
        profundidades_grafico = [] 
        diametros_grafico = []     
        
        for i in range(15):
            detectados_na_linha = np.where(matriz_deteccao[i, :] == 1)[0]
            if len(detectados_na_linha) > 0:
                menor_j = np.min(detectados_na_linha) 
                prof_limite = profundidades_x[menor_j]
                diam_atual = diametros_y[i]
                
                iqf += diam_atual * prof_limite
                profundidades_grafico.append(prof_limite)
                diametros_grafico.append(diam_atual)
        
        iqf_inv = (100 / iqf) if iqf > 0 else 0
        
        col_iqf1, col_iqf2 = st.columns(2)
        with col_iqf1:
            st.metric("IQF Padrão", f"{iqf:.2f}")
        with col_iqf2:
            st.metric("IQF Inverso", f"{iqf_inv:.2f}")
        
        st.caption(f"ℹ️ Total de Quadrados Válidos Detectados: {contagem_vistos} / 225")
        st.divider()
        
        st.write("**Curva Contraste-Detalhe (C-D)**")
        fig_grafico, ax_grafico = plt.subplots(figsize=(6, 4))
        
        if len(diametros_grafico) > 0:
            dados_ordenados = sorted(zip(profundidades_grafico, diametros_grafico))
            prof_plot, diam_plot = zip(*dados_ordenados)
            ax_grafico.plot(prof_plot, diam_plot, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        
        ax_grafico.set_xlabel("Profundidade / Contraste (mm)")
        ax_grafico.set_ylabel("Diâmetro / Resolução (mm)")
        ax_grafico.set_xlim(0, 8.5)
        ax_grafico.set_ylim(0, 8.5)
        ax_grafico.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig_grafico)
