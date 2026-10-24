"""
App Streamlit: Pokémon de Fogo ou de Água?

Fluxo: o usuário envia uma imagem -> extraímos as features (mesmo código do treino, em features.py)
-> os modelos (SVM, Random Forest, KNN e Gradient Boosting, treinados no notebook) devolvem a
probabilidade de Fogo -> mostramos a classe prevista e a probabilidade.

Arquivos necessários na mesma pasta (todos dentro do arquivos_app.zip gerado pelo notebook):
  features.py, config.json, os arquivos modelo_*.joblib
  e, opcionalmente, resultados.csv, ablacao.csv e robustez.csv (para a aba de métricas)
"""
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from features import extrair   # MESMO código de features usado no treino (evita divergência)

st.set_page_config(page_title="Pokémon: Fogo ou Água?", page_icon="🔥", layout="wide")



# ------------------------------------------------------------------
# Carregamento dos modelos (cache: carrega uma vez só, não a cada clique)
# ------------------------------------------------------------------
@st.cache_resource
def carregar():
    # config.json guarda o conjunto de features, os arquivos dos modelos e os limiares (F2)
    with open("config.json") as f:
        cfg = json.load(f)
    arquivos = cfg.get("modelos", {"SVM": "modelo_svm.joblib", "Random Forest": "modelo_rf.joblib"})
    modelos = {nome: joblib.load(arq) for nome, arq in arquivos.items()}
    return modelos, cfg


modelos, cfg = carregar()
CONJUNTO = cfg["conjunto"]
LIMIARES = cfg["limiares"]
NOMES_2 = list(modelos)

# Extensão opcional de 3 classes (Fogo, Água, Outro), gravada pelo notebook na seção 8
TRES = cfg.get("tres_classes")


@st.cache_resource
def carregar3(arquivos):
    return {nome: joblib.load(arq) for nome, arq in arquivos.items()}


modelos3 = carregar3(TRES["modelos"]) if TRES else {}
CLASSES3 = TRES["classes"] if TRES else []                    # ['Água', 'Fogo', 'Outro'] (ordem das colunas)
NOMES_3 = list(modelos3)

# Descrição das features REALMENTE usadas pelos modelos (vem do config.json, escrito pelo notebook)
BLOCOS = {
    "cor": "cor (histogramas de matiz, saturação e brilho, e cores médias)",
    "cor2": "cores extras (espaço LAB, matiz médio e 3 cores dominantes)",
    "hog": "forma (HOG)",
}
_partes = [BLOCOS[n] for n in CONJUNTO.split("+") if n in BLOCOS]
DESCRICAO_FEATURES = (", ".join(_partes[:-1]) + " e " + _partes[-1]) if len(_partes) > 1 else _partes[0]

# ------------------------------------------------------------------
# Barra lateral: quais modelos mostrar e como decidir entre Fogo e Água
# ------------------------------------------------------------------
MODO3 = False
if TRES:
    st.sidebar.header("🧭 Tipo de classificação")
    MODO3 = st.sidebar.radio("Classes", ["2 classes: Fogo ou Água", "3 classes: Fogo, Água ou Outro"]).startswith("3")
NOMES = NOMES_3 if MODO3 else NOMES_2

st.sidebar.header("🤖 Modelos")
SELECIONADOS = st.sidebar.multiselect("Mostrar", NOMES, default=NOMES, key=f"sel_{MODO3}")

modo = "maior probabilidade"
LIMIAR_ATUAL = {}
PESOS3 = {}
if MODO3:
    PESOS_NB = TRES.get("pesos_outro", {})                    # pesos de 'Outro' escolhidos no notebook (F1 macro)
    st.sidebar.header("⚙️ Decisão (3 classes)")
    opcoes3 = ["Maior probabilidade"] + (["Ajustada (peso de Outro do notebook)"] if PESOS_NB else []) + ["Personalizada"]
    modo = st.sidebar.radio("Como decidir?", opcoes3)
    if modo.startswith("Ajustada"):
        PESOS3 = {nome: float(PESOS_NB.get(nome, 1.0)) for nome in NOMES}
    elif modo.startswith("Personalizada"):
        w3 = st.sidebar.slider("Peso de Outro", 0.05, 1.0, 1.0, 0.05)
        PESOS3 = {nome: w3 for nome in NOMES}
    else:
        PESOS3 = {nome: 1.0 for nome in NOMES}
    st.sidebar.caption(
        "A probabilidade de Outro é multiplicada por esse peso antes de escolher a classe. "
        "Peso menor = o modelo diz Outro com menos facilidade (mais recall de Fogo e Água, mais alarmes falsos)."
    )
else:
    st.sidebar.header("⚙️ Limiar de decisão")
    modo = st.sidebar.radio("Como decidir?", ["Ajustado para F2 (notebook)", "Padrão (0,50)", "Personalizado"])
    if modo.startswith("Ajustado"):
        LIMIAR_ATUAL = {nome: LIMIARES.get(nome, 0.50) for nome in NOMES}   # limiares que maximizam o F2
    elif modo.startswith("Padrão"):
        LIMIAR_ATUAL = {nome: 0.50 for nome in NOMES}                       # corte clássico de 0,5
    else:
        lim = st.sidebar.slider("Limiar", 0.05, 0.95, 0.50, 0.01)
        LIMIAR_ATUAL = {nome: lim for nome in NOMES}
    st.sidebar.caption(
        "Limiar baixo = mais recall de Fogo, mas mais alarmes falsos (Água classificado como Fogo). "
        "Limiar alto = o contrário."
    )


@st.cache_resource
def carregar_pokedex():
    # {número: nome} gerado pelo notebook (pokedex.json); vazio se o arquivo não existir
    if not os.path.exists("pokedex.json"):
        return {}
    with open("pokedex.json", encoding="utf-8") as f:
        return json.load(f)


def nome_do_arquivo(nome_arquivo):
    # Descobre o Pokémon pelo nome do arquivo: número da Pokédex (099.png) ou nome (kingler.png).
    # Devolve (número, nome) ou None. Não olha a imagem: é só o nome do arquivo.
    dex = carregar_pokedex()
    if not dex:
        return None
    base = os.path.splitext(os.path.basename(nome_arquivo))[0]
    achou = re.search(r"\d+", base)
    if achou and str(int(achou.group())) in dex:
        n = str(int(achou.group()))
        return int(n), dex[n].replace("-", " ").title()
    chave = re.sub(r"[^a-z0-9]", "", base.lower())
    for n, nome in dex.items():
        if re.sub(r"[^a-z0-9]", "", nome.lower()) == chave:
            return int(n), nome.replace("-", " ").title()
    return None


def probabilidades(x):
    # P(Fogo) de cada modelo
    return {nome: float(m.predict_proba(x)[0, 1]) for nome, m in modelos.items()}


def probabilidades3(x):
    # Vetor [P(Água), P(Fogo), P(Outro)] de cada modelo
    return {nome: m.predict_proba(x)[0] for nome, m in modelos3.items()}


def decisao(nome, p_fogo):
    classe = "Fogo" if p_fogo >= LIMIAR_ATUAL[nome] else "Água"   # usa o limiar escolhido na barra lateral
    conf = p_fogo if classe == "Fogo" else 1 - p_fogo             # probabilidade da classe prevista
    return classe, conf


# ------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------
if MODO3:
    st.title("🔥💧❔ Fogo, Água ou outro tipo?")
    TIPO_CLASSIF = "Extensão de 3 classes (Fogo, Água, Outro)"
else:
    st.title("🔥💧 Esse Pokémon é de Fogo ou de Água?")
    TIPO_CLASSIF = "Classificação binária"
st.caption(TIPO_CLASSIF + " de imagens com modelos clássicos (SVM, Random Forest, KNN, Gradient Boosting) "
           f"usando features de {DESCRICAO_FEATURES}")

aba_pred, aba_metricas, aba_sobre = st.tabs(["🔍 Classificar imagem", "📊 Métricas dos modelos", "ℹ️ Sobre"])

with aba_pred:
    arquivo = st.file_uploader("Envie a imagem de um Pokémon (PNG ou JPG)", type=["png", "jpg", "jpeg"])
    st.caption("Dica: PNG com fundo transparente funciona melhor. Fundo branco também é ignorado; "
               "outros fundos entram nas features de cor e podem atrapalhar.")

    if arquivo is None:
        st.info("Envie uma imagem para ver a previsão.")
    elif not SELECIONADOS:
        st.warning("Escolha pelo menos um modelo na barra lateral.")
    else:
        try:
            img = Image.open(arquivo)
            x = extrair(img, CONJUNTO).reshape(1, -1)    # 1 imagem, N features
            probs = probabilidades3(x) if MODO3 else probabilidades(x)
        except Exception as erro:
            st.error(f"Não consegui processar essa imagem: {erro}")
            st.stop()

        col_img, col_res = st.columns([1, 2])
        with col_img:
            st.image(img, caption="Imagem enviada", width="stretch")
            achado = nome_do_arquivo(arquivo.name)
            if achado:
                st.subheader(f"🔎 #{achado[0]:03d} {achado[1]}")
                st.caption("Nome pelo número ou nome do arquivo (Pokédex nacional). Não é identificado pela imagem.")

        with col_res:
            # Até 3 modelos por linha, para os cartões não ficarem estreitos
            for i in range(0, len(SELECIONADOS), 3):
                colunas = st.columns(3)
                for col, nome in zip(colunas, SELECIONADOS[i:i + 3]):
                    if MODO3:
                        ajustada = probs[nome] * np.array([1.0, 1.0, PESOS3[nome]])   # reduz só a coluna 'Outro'
                        idx_classe = int(np.argmax(ajustada))
                        classe = CLASSES3[idx_classe]
                        conf = float(probs[nome][idx_classe])                         # probabilidade original da classe
                    else:
                        classe, conf = decisao(nome, probs[nome])
                    with col:
                        # Título com altura fixa (2 linhas): evita que "Gradient Boosting" desalinhe os cartões
                        st.markdown(
                            f"<div style='font-size:1.35rem;font-weight:600;line-height:1.6rem;"
                            f"height:3.3rem;overflow:hidden'>{nome}</div>",
                            unsafe_allow_html=True,
                        )
                        if classe == "Fogo":
                            st.error("🔥 Fogo")
                        elif classe == "Água":
                            st.info("💧 Água")
                        else:
                            st.warning("❔ Outro")
                        st.metric("Probabilidade da classe prevista", f"{conf:.2f}")
                        if MODO3:
                            for c, v in zip(CLASSES3, probs[nome]):
                                st.progress(float(v), text=f"P({c}) = {v:.2f}")
                        else:
                            st.progress(probs[nome], text=f"P(Fogo) = {probs[nome]:.2f}  |  limiar = {LIMIAR_ATUAL[nome]:.2f}")

            if MODO3:
                st.caption(
                    f"Modo de 3 classes (extensão). Decisão: **{modo}**. "
                    "\"Outro\" reúne todos os Pokémon que não são de Fogo nem de Água."
                )
            else:
              st.caption(
                f"Limiar em uso: **{modo}**. O limiar do notebook foi ajustado para maximizar o **F2** "
                "(prioriza o recall de Fogo). Mude o limiar na barra lateral para ver o trade-off. "
                "Uma probabilidade perto do limiar indica um caso duvidoso."
            )

with aba_metricas:
    st.subheader("Desempenho (validação cruzada estratificada de 5 folds)")
    tabelas = [
        ("resultados.csv", "**Modelos ajustados: limiar 0,50 vs limiar ajustado para F2**"),
        ("ablacao.csv", "**Ablação de features (SVM e Random Forest, limiar 0,50)**"),
        ("robustez.csv", "**Teste de robustez: imagem original vs variantes (shiny e tons de cinza)**"),
        ("robustez_limiar.csv", "**Robustez: o efeito do limiar (F2, 0,50 e oráculo)**"),
        ("tres_classes.csv", "**Extensão de 3 classes (Fogo, Água, Outro), decisão pela maior probabilidade**"),
        ("tres_classes_peso.csv", "**Extensão de 3 classes: sem ajuste (peso 1,00) vs peso de Outro ajustado (F1 macro)**"),
    ]
    achou = False
    for arq, titulo in tabelas:
        try:
            tabela = pd.read_csv(arq)
        except FileNotFoundError:
            continue
        achou = True
        st.markdown(titulo)
        st.dataframe(tabela, width="stretch", hide_index=True)
    if not achou:
        st.info("Coloque resultados.csv, ablacao.csv e robustez.csv (exportados pelo notebook) na pasta do app.")

with aba_sobre:
    st.markdown(
        f"""
**Dados:** Pokemon Images and Types (Kaggle). Ficamos com os Pokémon de tipo Fogo ou Água
(no tipo 1 ou no tipo 2) e descartamos os que têm os dois.

**Features:** {DESCRICAO_FEATURES}, calculadas só nos pixels do Pokémon (o fundo transparente ou branco é
ignorado). Conjunto usado neste app: **{CONJUNTO}**. No notebook, a ablação comparou cor, forma (HOG) e cores
extras, e o conjunto com melhor ROC AUC médio foi o usado nos modelos finais.

**Modelos:** {", ".join(modelos)}, com parâmetros ajustados por busca em grade e validação aninhada.

**Avaliação:** validação cruzada estratificada (5 folds), porque o conjunto é pequeno. O teste de robustez
treina com a imagem original e testa em variantes shiny e em tons de cinza de espécies não vistas.

**Extensão de 3 classes:** a classe "Outro" reúne os Pokémon que não são de Fogo nem de Água. É um problema
mais difícil, porque "Outro" junta muitos tipos e é a classe majoritária; o modo de 3 classes do app usa os
modelos treinados nessa extensão.

**Limitações:** Pokémon da mesma evolução são parecidos e podem deixar o resultado otimista; Pokémon com cores
que não combinam com o tipo são difíceis; no modo de 2 classes, um Pokémon de outro tipo é forçado a cair em
Fogo ou Água.
        """
    )
