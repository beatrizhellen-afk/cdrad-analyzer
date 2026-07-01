import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

st.set_page_config(page_title="CDRAD Analyzer Pro", layout="wide")

st.title("CDRAD Analyzer Pro ⚛️")
st.write("Software de análise automatizada do fantoma CDRAD para otimização de doses.")

# Valores padrão do CDRAD (Diâmetros e Profundidades em mm)
dimensoes = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 8.0]

# --- MENU LATERAL ---
st.sidebar.header("🔧 Controles da Imagem")

st.sidebar.subheader("1. Correção de Orientação e Alinhamento")
espelhar_h = st.sidebar.checkbox("↔️ Espelhar Horizontalmente")
espelhar_v = st.sidebar.checkbox("↕️ Espelhar Verticalmente")

angulo_rotacao = st.sidebar.number_input(
    "🔄 Alinhamento Fino (Graus)", 
    min_value=-180.0, 
    max_value=180.0, 
    value=0.0, 
    step=0.1, 
    format="%.1f"
)
st.sidebar.caption("Use os botões de + e - para girar a imagem passo a passo.")

st.sidebar.subheader("2. Recorte da Matriz (ROI)")
st.sidebar.write("Ajuste as barras para isolar apenas os furos da matriz.")
margem_sup = st.sidebar.slider("Cortar Topo (%)", 0, 40, 10)
margem_inf = st.sidebar.slider("Cortar Base (%)", 0, 40, 10)
margem_esq = st.sidebar.slider("Cortar Esquerda (%)", 0, 40, 10)
margem_dir = st.sidebar.slider("Cortar Direita (%)", 0, 40, 10)

st.sidebar.subheader("3. Sensibilidade")
sensibilidade = st.sidebar.slider("Limiar de Detecção (Ruído)", 5, 30, 12)

arquivo_dicom = st.file_uploader("Faça o upload da imagem DICOM do CDRAD (.dcm)", type=["dcm"])

if arquivo_dicom is not None:
    dcm = pydicom.dcmread(arquivo_dicom)
    imagem_bruta = dcm.pixel_array
    
    if espelhar_h:
        imagem_bruta = np.fliplr(imagem_bruta)
    if espelhar_v:
        imagem_bruta = np.flipud(imagem_bruta)
    
    imagem_norm = cv2.normalize(imagem_bruta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if angulo_rotacao != 0.0:
        h_orig, w_orig = imagem_norm.shape
        centro_img = (w_orig // 2, h_orig // 2)
        M = cv2.getRotationMatrix2D(centro_img, angulo_rotacao, 1.0)
        imagem_processada = cv2.warpAffine(imagem_norm, M, (w_orig, h_orig), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        imagem_processada = imagem_norm.copy()
        h_orig, w_orig = imagem_norm.shape
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imagem_realcada = clahe.apply(imagem_processada)

    # Recorte (ROI)
    h_img, w_img = imagem_realcada.shape
    top = int(h_img * (margem_sup / 100.0))
    bottom = int(h_img * (1 - (margem_inf / 100.0)))
    left = int(w_img * (margem_esq / 100.0))
    right = int(w_img * (1 - (margem_dir / 100.0)))
    imagem_roi = imagem_realcada[top:bottom, left:right]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Análise da Grade CDRAD (Alinhada)")
        h, w = imagem_roi.shape
        cell_h, cell_w = h // 15, w // 15
        contagem_vistos = 0
        matriz_deteccao = np.zeros((15, 15))
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(imagem_roi, cmap='gray')
        
        for i in range(15): 
            for j in range(15): 
                y, x = i * cell_h, j * cell_w
                roi_celula = imagem_roi[y:y+cell_h, x:x+cell_w]
                
                if np.std(roi_celula) > sensibilidade: 
                    matriz_deteccao[i, j] = 1
                    contagem_vistos += 1
                    rect = plt.Rectangle((x, y), cell_w, cell_h, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
                else:
                    rect = plt.Rectangle((x, y), cell_w, cell_h, linewidth=1, edgecolor='r', facecolor='none', alpha=0.2)
                ax.add_patch(rect)
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        st.subheader("📊 Métricas da Imagem")
        st.metric("Quadrados Detectados", f"{contagem_vistos} / 225")
        
        indices_detectados = np.where(matriz_deteccao == 1)
        if len(indices_detectados[1]) > 0:
            menor_dia = dimensoes[np.min(indices_detectados[1])] 
            menor_prof = dimensoes[np.min(indices_detectados[0])] 
        else:
            menor_dia, menor_prof = 0, 0

        met_res_esp, met_threshold = st.columns(2)
        with met_res_esp:
            st.metric("Resolução Espacial Limite", f"{menor_dia} mm")
        with met_threshold:
            st.metric("Threshold Baixo Contraste", f"{menor_prof} mm")
        
        st.divider()
        st.subheader("Índice de Qualidade")
        
        iqf = 0
        for j in range(15):
            detectados_na_coluna = np.where(matriz_deteccao[:, j] == 1)[0]
            if len(detectados_na_coluna) > 0:
                prof_limite = dimensoes[np.min(detectados_na_coluna)]
                iqf += dimensoes[j] * prof_limite
        
        st.title(f"IQF Padrão: {iqf:.2f}")
        st.info("**Interpretação:** Quanto MENOR o IQF, melhor a qualidade.")
        st.divider()

        # --- NOVA FUNCIONALIDADE: GERADOR DE RELATÓRIO DO MUNDO REAL ---
        # Aqui estruturamos o texto que vai para o arquivo final salvável
        data_atual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        fabricante = dcm.get('Manufacturer', 'Desconhecido')
        kv_dicom = dcm.get('KVP', 'Não informado no DICOM')
        mas_dicom = dcm.get('XRayTubeCurrentInuA', 'Não informado no DICOM') # Ou Exposure
        
        conteudo_relatorio = f"""================================================================
          RELATÓRIO DE QUALIDADE DE IMAGEM & OTIMIZAÇÃO DE DOSE
                       CDRAD ANALYZER PRO v1.8
================================================================
Data/Hora da Análise: {data_atual}
Equipamento Analisado (Fabricante): {fabricante}
Resolução da Imagem: {w_orig} x {h_orig} pixels
Parâmetros DICOM Extraídos:
 - Voltagem da Ampola (kVp): {kv_dicom}
 - Corrente/Tempo (mAs): {mas_dicom}

----------------------------------------------------------------
PARÂMETROS DE CONFIGURAÇÃO DO SOFTWARE (SETUP DA ROI):
----------------------------------------------------------------
- Espelhamento Horizontal aplicado: {espelhar_h}
- Espelhamento Vertical aplicado: {espelhar_v}
- Rotação Fina de Alinhamento: {angulo_rotacao} graus
- Margens de Recorte ROI (Topo/Base/Esq/Dir): {margem_sup}% / {margem_inf}% / {margem_esq}% / {margem_dir}%
- Limiar de Sensibilidade algoritmo (STD): {sensibilidade}

----------------------------------------------------------------
MÉTRICAS RESULTANTES (FÍSICA MÉDICA):
----------------------------------------------------------------
- Total de Quadrados Úteis Detectados: {contagem_vistos} / 225
- Resolução Espacial Limite (Menor Detalhe): {menor_dia} mm
- Threshold de Baixo Contraste (Menor Profundidade): {menor_prof} mm

================================================================
ÍNDICE DE QUALIDADE DE IMAGEM RESULTANTE:
>>> INVERSE IMAGE QUALITY FIGURE (IQF): {iqf:.2f} <<<
================================================================
Nota Técnica: Valores menores de IQF indicam melhor desempenho 
de baixo contraste e resolução espacial combinados.

Responsável Técnico: _________________________________________
Físico Médico / Supervisor de Radioproteção
"""

        # Substituindo o botão comum pelo botão de Download real do Streamlit
        st.download_button(
            label="📥 Descarregar Relatório de Otimização (.txt)",
            data=conteudo_relatorio,
            file_name=f"relatorio_otimizacao_cdrad_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )
