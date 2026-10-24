# Pokémon: Fogo ou Água? Classificação binária de imagens com ML clássico

Projeto da pós-graduação em IA. Dada a imagem de um Pokémon, os modelos dizem se ele é do tipo **Fogo** ou **Água**
e devolvem a **probabilidade** da classe prevista (ex.: `Fogo, 0,81`).

- **Notebook (Google Colab):** treina e avalia os modelos, e exporta os arquivos do app.
- **App (Streamlit):** envia uma imagem e mostra a previsão de cada modelo.

> Projeto educacional, sem fins de diagnóstico ou de uso real. Pokémon é marca de seus respectivos donos; as imagens
> pertencem a seus detentores e não estão incluídas neste repositório.

---

## Requisitos do trabalho e onde cada um está

| Requisito | Onde |
|---|---|
| Classificação binária com imagens | Fogo (1) vs Água (0) |
| Saída em probabilidade | `predict_proba`: seção 5 do notebook e cartões do app |
| Métricas: acurácia, precisão, recall, F1, F2 | Tabelas das seções 4 e 6 do notebook; aba "Métricas" do app |
| Pelo menos 2 modelos (KNN, Random Forest, Gradient Boosting, SVM) | Os quatro |
| Matriz de confusão | Seções 4 e 6 do notebook |

---

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `classificacao_pokemon_fogo_agua.ipynb` | Notebook do Colab: dados, features, modelos, métricas, robustez e exportação |
| `app.py` | App Streamlit |
| `requirements.txt` | Dependências do app |
| `features.py` | Extração de features. **É gerado pelo notebook** (dentro do `arquivos_app.zip`) |
| `arquivos_app.zip` | Gerado pelo notebook: modelos `.joblib` (2 e 3 classes), `config.json`, tabelas `.csv` e `features.py` |

O `features.py` é compartilhado entre o notebook e o app de propósito: o app usa **exatamente o mesmo código** de
features do treino, então as duas partes nunca ficam diferentes.

---

## Como rodar

### 1. Treinar no Colab

1. Abra `classificacao_pokemon_fogo_agua.ipynb` no [Google Colab](https://colab.research.google.com/).
2. Execute todas as células, na ordem (*Ambiente de execução → Executar tudo*).
   - Baixa os dados direto do Kaggle pelo `kagglehub`.
   - A seção 7 (teste de robustez) baixa um segundo dataset de cerca de 785 MB.
   - A validação cruzada aninhada (seção 6) leva alguns minutos.
3. A última célula baixa o **`arquivos_app.zip`**.

### 2. Rodar o app no seu computador

```bash
# 1) Coloque na mesma pasta: app.py, requirements.txt e o conteúdo do arquivos_app.zip (extraído)
# 2) Instale as dependências
pip install -r requirements.txt

# 3) Rode o app
streamlit run app.py
```

> **Atenção à versão do scikit-learn.** Os modelos `.joblib` só carregam com a mesma versão do scikit-learn usada no
> Colab. A última célula do notebook imprime essa versão. Se o app der erro ao carregar os modelos, instale a mesma:
> `pip install scikit-learn==<versão>`.

---

## Dados

| Fonte | Uso |
|---|---|
| [Pokemon Images and Types](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types) (Kaggle) | Treino e avaliação: uma imagem por Pokémon e um CSV com os tipos |
| [Pokemon Dataset 1000](https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000) (Kaggle) | Só no teste de robustez: cerca de 26 imagens por espécie (sprites antigos, arte oficial, variantes shiny e em tons de cinza) |

**Como virou um problema binário:** ficam os Pokémon com o tipo Fogo **ou** o tipo Água (no tipo 1 ou no tipo 2).
Quem tem os dois tipos ao mesmo tempo, ou nenhum, sai do conjunto. Isso deixa o conjunto **pequeno** e **desbalanceado**
(há mais Pokémon de Água que de Fogo).

---

## Método

### Features (`features.py`)

Os modelos clássicos não recebem imagens, e sim uma tabela de números. Cada imagem de 128x128 vira um vetor com:

- **`cor`**: histogramas de matiz, saturação e brilho (HSV), mais cores médias, calculados só nos pixels do Pokémon.
- **`cor2`**: espaço LAB, matiz médio (circular) e as 3 cores dominantes (k-means).
- **`hog`**: histograma de gradientes orientados, que descreve a forma pelas bordas.

O fundo transparente ou branco é ignorado. A **ablação** compara cinco conjuntos (`cor`, `hog`, `cor+hog`, `cor+cor2`,
`cor+cor2+hog`) para mostrar o que realmente ajuda. O conjunto com maior ROC AUC médio (SVM e Random Forest) foi
**`cor`**, então **os modelos finais e o app usam só as features de cor (31 números)**. A forma (HOG) e as cores
extras não melhoraram o resultado.

### Modelos

SVM (kernel RBF), Random Forest, KNN e Gradient Boosting. Parâmetros
ajustados com `GridSearchCV`. Onde o modelo permite, `class_weight='balanced'` trata o desbalanceamento.

### Avaliação

- **Validação cruzada estratificada de 5 folds**, porque o conjunto é pequeno e um teste separado teria poucas
  dezenas de imagens, com métricas instáveis. Cada imagem é prevista por um modelo que nunca a viu.
- **Validação aninhada:** o ajuste de parâmetros roda *dentro* de cada fold, então os parâmetros nunca são escolhidos
  olhando a imagem que está sendo avaliada.
- **Métricas:** acurácia, precisão, recall, F1, **F2**, ROC AUC, matriz de confusão e curva ROC.
- **Limiar de decisão:** além do corte padrão de 0,50, usamos o limiar que maximiza o **F2**. O F2 pesa o recall duas
  vezes mais que a precisão, então o modelo prefere um alarme falso a deixar um Pokémon de Fogo passar. Como o
  problema é lúdico, esse custo é ilustrativo e serve para mostrar o trade-off.

### Teste de robustez

Treina com a imagem **original** das espécies de treino e testa nas **variantes** (shiny e tons de cinza) das espécies
de teste, sempre espécies não vistas pelo modelo. Mede o quanto o modelo depende da cor.

---

## Extensão: 3 classes (Fogo, Água, Outro)

O modelo binário só conhece Fogo e Água, então um Pokémon de outro tipo (ex.: Graveler, de Pedra) é forçado a cair em
uma das duas. A seção 8 do notebook testa uma **extensão com a classe "Outro"**, usando todos os Pokémon do dataset
(menos os que são Fogo e Água ao mesmo tempo).

- **Mais difícil:** "Outro" junta os outros 16 tipos e é a classe majoritária, então a acurácia sozinha engana. Avaliamos
  com acurácia balanceada, F1 macro e o recall de cada classe.
- **Decisão:** pela maior probabilidade (sem o limiar do F2).
- **No app:** a barra lateral tem a opção "3 classes: Fogo, Água ou Outro", com as três probabilidades por modelo.
- **Resultado principal do trabalho:** continua sendo o binário (o enunciado pede classificação binária); esta parte é uma extensão.
- **Resultados da extensão:** veja a seção abaixo.

### Resultados da extensão (validação cruzada estratificada de 5 folds, decisão pela maior probabilidade)

"Outro" é a classe majoritária: o chute "sempre Outro" acerta cerca de **0,76** (estimado a partir das métricas;
confira a linha `Chute "sempre Outro"` impressa na seção 8 do notebook). Por isso a acurácia sozinha engana.

| Modelo | Acurácia | Acurácia balanceada | F1 macro | Recall Água | Recall Fogo | Recall Outro | Precisão Água | Precisão Fogo | Precisão Outro |
|---|---|---|---|---|---|---|---|---|---|
| SVM | 0,778 | 0,480 | 0,520 | 0,238 | 0,254 | 0,946 | 0,574 | 0,593 | 0,801 |
| Random Forest | 0,736 | 0,672 | 0,613 | 0,608 | 0,635 | 0,774 | 0,485 | 0,381 | 0,881 |
| KNN | 0,800 | 0,516 | 0,559 | 0,377 | 0,222 | 0,948 | 0,662 | 0,538 | 0,823 |
| Gradient Boosting | 0,771 | 0,584 | 0,595 | 0,469 | 0,413 | 0,872 | 0,545 | 0,433 | 0,843 |

**Exemplos (probabilidades do SVM, previsões de validação cruzada; "ajustada" usa o peso 0,40 de Outro):**

| Pokémon | Real | Água | Fogo | Outro | Previsto (maior prob.) | Previsto (ajustada) |
|---|---|---|---|---|---|---|
| Bulbasaur | Outro | 0,04 | 0,00 | 0,96 | Outro | Outro |
| Graveler | Outro | 0,08 | 0,01 | 0,91 | Outro | Outro |
| Pikachu | Outro | 0,08 | 0,05 | 0,88 | Outro | Outro |
| Vaporeon | Água | 0,58 | 0,01 | 0,42 | Água | Água |
| Charizard | Fogo | 0,18 | 0,17 | 0,64 | Outro | Outro |
| Kingler | Água | 0,19 | 0,34 | 0,47 | Outro | Fogo |
| Magikarp | Água | 0,14 | 0,43 | 0,43 | Fogo ou Outro (empate) | Fogo |

Os exemplos foram escolhidos a dedo e não representam o desempenho geral.

**Conclusões da extensão**
- **A acurácia (0,74 a 0,80) quase não passa do chute de ~0,76**; o Random Forest (0,736) fica abaixo dele. As métricas
  que valem são a acurácia balanceada (o acaso seria 0,33) e o F1 macro.
- **O Random Forest é o melhor para as classes pequenas** (acurácia balanceada 0,672; recall de 0,61 em Água e 0,64 em
  Fogo), ao custo de precisão baixa em Fogo (0,38) e Água (0,49).
- **SVM e KNN quase só dizem "Outro"** (recall de Outro ~0,95; recall de Fogo 0,22 a 0,25).
- **O recall de Fogo e Água cai muito em relação ao binário** (de ~0,8 para 0,22 a 0,64). Possíveis causas, não
  testadas: "Outro" tem muitos Pokémon vermelhos, laranjas e azuis; é ~3 vezes maior que as duas classes juntas; e usamos só cor.
- **O Graveler passa a ser reconhecido como "Outro"** (0,91 no SVM), o caso que motivou a extensão. Em troca, alguns Fogo e
  Água reais, como o Charizard e o Kingler, viram "Outro", porque a classe maior puxa a decisão.
- **Limitações:** os parâmetros vieram do ajuste binário (sem nova busca); não há intervalo de confiança; evoluções
  parecidas entre si podem deixar o resultado otimista (por exemplo, Geodude e Golem para o Graveler).

### Ajuste da decisão: peso para "Outro"

A probabilidade de "Outro" é multiplicada por um peso entre 0,05 e 1,00 antes de escolher a classe. O peso de cada
modelo é o que maximiza o F1 macro. Ele foi escolhido nas mesmas previsões de validação cruzada usadas para avaliar,
então o ganho é um pouco otimista.

| Modelo | Peso de Outro | Acurácia | Acur. balanceada | F1 macro | Recall Água | Recall Fogo | Recall Outro |
|---|---|---|---|---|---|---|---|
| SVM | 1,00 | 0,778 | 0,480 | 0,520 | 0,238 | 0,254 | 0,946 |
| SVM | 0,40 | 0,772 | 0,645 | 0,630 | 0,554 | 0,540 | 0,842 |
| KNN | 1,00 | 0,800 | 0,516 | 0,559 | 0,377 | 0,222 | 0,948 |
| KNN | 0,55 | 0,776 | 0,592 | 0,601 | 0,508 | 0,397 | 0,872 |

Random Forest e Gradient Boosting ficaram com peso 1,00 (o ajuste não melhorou o F1 macro deles).

- **O ajuste ajuda bastante os modelos que quase só diziam "Outro":** o F1 macro do SVM vai de 0,52 a 0,63
  (melhor F1 macro da extensão) e o do KNN de 0,56 a 0,60.
- **O custo aparece na acurácia** (no SVM cai de 0,778 para 0,772 e no KNN de 0,800 para 0,776) e na precisão de Água e Fogo (~0,50),
  porque mais Pokémon de "Outro" passam a ser classificados como Fogo ou Água.
- **Nos exemplos, o ajuste não recupera nenhum caso:** o Charizard continua "Outro" (0,64 x 0,40 = 0,26, ainda maior
  que 0,18), o Kingler passa de "Outro" para "Fogo" e o Magikarp continua "Fogo", todos errados. Sem o ajuste, o SVM
  manda 10 Pokémon de "Outro" para Fogo.

---

## O app

- Envie um PNG (de preferência com fundo transparente) ou JPG.
- Veja a classe e a probabilidade de cada modelo.
- Na barra lateral: escolha quais modelos mostrar e o **limiar de decisão** (ajustado para F2, padrão 0,50 ou
  personalizado) para ver o trade-off ao vivo.
- Abas **Métricas dos modelos** e **Sobre**.

Sugestão de demo: o Kingler (Água, mas laranja) e o Graveler (Pedra/Terra, um tipo que o modelo não conhece).
Troque o limiar na barra lateral e veja como a decisão de cada modelo muda.

---

## Limitações

- **Só existem duas classes.** Um Pokémon de outro tipo (ex.: Graveler, de Pedra) é forçado a cair em Fogo ou Água.
- **A cor domina.** Pokémon cuja cor não combina com o tipo (Água vermelho, Fogo azul) são difíceis.
- **Resultado possivelmente otimista:** evoluções do mesmo Pokémon são parecidas e podem cair em folds diferentes; o
  conjunto de features e o limiar foram escolhidos nas mesmas previsões usadas para avaliar.
- **Robustez:** o rótulo das variantes vem da espécie, e não é possível marcar quais imagens são shiny; só separamos
  automaticamente as em tons de cinza. O dataset mistura estilos (pixel art e arte oficial), o que também afeta o resultado.
- **Fundo:** o modelo foi treinado em imagens com fundo transparente. Fotos e desenhos com fundo colorido entram nas
  features de cor e podem atrapalhar.

## Ideias para continuar

- Dividir a validação por **linha evolutiva**, para reduzir o otimismo.
- Embeddings de uma CNN pré-treinada como features, se o enunciado permitir.

---

## Resultados

Conjunto de **193 Pokémon** (cerca de 63 de Fogo e 130 de Água, estimado a partir das métricas; confira a contagem
impressa na seção 2 do notebook). Um chute "sempre Água" teria cerca de **0,67 de acurácia**. Avaliação por validação
cruzada estratificada de 5 folds; modelos ajustados com validação aninhada; features: `cor`.

### Modelos ajustados (limiar 0,50 e limiar ajustado para F2)

| Modelo | Limiar | Acurácia | Precisão | Recall | F1 | F2 | ROC AUC |
|---|---|---|---|---|---|---|---|
| SVM | 0,50 | 0,839 | 0,735 | 0,794 | 0,763 | 0,781 | 0,901 |
| SVM | 0,26 | 0,839 | 0,682 | 0,952 | 0,795 | 0,882 | 0,901 |
| Random Forest | 0,50 | 0,845 | 0,739 | 0,810 | 0,773 | 0,794 | 0,913 |
| Random Forest | 0,34 | 0,829 | 0,667 | 0,952 | 0,784 | 0,877 | 0,913 |
| KNN | 0,50 | 0,834 | 0,712 | 0,825 | 0,765 | 0,800 | 0,889 |
| KNN | 0,29 | 0,803 | 0,637 | 0,921 | 0,753 | 0,845 | 0,889 |
| Gradient Boosting | 0,50 | 0,839 | 0,750 | 0,762 | 0,756 | 0,759 | 0,900 |
| Gradient Boosting | 0,08 | 0,782 | 0,608 | 0,937 | 0,738 | 0,845 | 0,900 |

### Ablação de features (SVM e Random Forest, limiar 0,50)

| Features | SVM (ROC AUC) | Random Forest (ROC AUC) |
|---|---|---|
| `cor` | 0,895 | 0,911 |
| `hog` | 0,627 | 0,689 |
| `cor+hog` | 0,833 | 0,903 |
| `cor+cor2` | 0,896 | 0,903 |
| `cor+cor2+hog` | 0,863 | 0,905 |

### Teste de robustez (ROC AUC e acurácia, limiar ajustado para F2)

| Modelo | AUC original | AUC variantes | Acurácia original | Acurácia variantes |
|---|---|---|---|---|
| SVM | 0,901 | 0,826 | 0,839 | 0,746 |
| Random Forest | 0,913 | 0,846 | 0,829 | 0,618 |
| KNN | 0,889 | 0,825 | 0,803 | 0,648 |
| Gradient Boosting | 0,900 | 0,817 | 0,782 | 0,472 |

Variantes: 1.158 imagens (6 por espécie). Em tons de cinza há só 37 imagens, pouco para conclusões firmes.

### Robustez: é o limiar ou é a imagem?

Acurácia nas variantes com o limiar do F2, com 0,50 e com o "melhor limiar das variantes" (oráculo, pelo F1; otimista,
usa os rótulos das variantes e serve só de diagnóstico). Chute "sempre Água": 0,674.

| Modelo | Original (0,50) | Variantes (limiar F2) | Variantes (0,50) | Variantes (oráculo) | % previsto Fogo (limiar F2, variantes) |
|---|---|---|---|---|---|
| SVM | 0,839 | 0,746 | 0,793 | 0,775 | 45,9% |
| Random Forest | 0,845 | 0,618 | 0,751 | 0,800 | 65,6% |
| KNN | 0,834 | 0,648 | 0,756 | 0,756 | 60,3% |
| Gradient Boosting | 0,839 | 0,472 | 0,702 | 0,762 | 85,2% |

Nas imagens originais, a % prevista como Fogo fica em 45% a 50% com o limiar do F2 (a proporção real de Fogo é de ~33%).

### Conclusões

- **A cor carrega a informação.** Só com HOG, a acurácia (0,67) é igual ao chute e o SVM quase nunca diz "Fogo". Só com
  cor, o ROC AUC é de 0,90 a 0,91.
- **Juntar o HOG piorou o SVM** (AUC 0,895 → 0,833), possivelmente porque as 324 features de forma diluem as 31 de cor.
  As cores extras (LAB e cores dominantes) não trouxeram ganho.
- **O ajuste de parâmetros quase não mudou nada** (AUC +0,006 no SVM e +0,002 no Random Forest).
- **O Random Forest teve o maior ROC AUC** (0,913, contra 0,901 do SVM). A diferença é pequena e não há
  intervalo de confiança.
- **O limiar do F2 troca precisão por recall.** No Random Forest, de 0,50 para 0,34 o recall sobe de 0,810 para 0,952 e a
  precisão cai de 0,739 para 0,667.
- **Robustez:** o ROC AUC cai pouco nas variantes (de ~0,90 para ~0,85), mas o limiar ajustado nas imagens originais
  **não transfere**: Gradient Boosting, Random Forest e KNN passam a prever Fogo em 60% a 85% das variantes
  (contra ~33% de Fogo de verdade), e a acurácia cai, ficando abaixo do chute de 0,674 no Random Forest e no Gradient
  Boosting. O SVM é o único estável. O F2 sozinho esconde isso, porque o recall continua alto.
- **Recalibrar o limiar recupera a maior parte da queda, mas não toda.** Com 0,50, a acurácia nas variantes sobe para
  0,70 a 0,79. No Random Forest, de uma queda total de 0,21, o limiar explica 0,13 (cerca de dois terços); sobram ~0,08 de
  acurácia que o limiar não explica e que mostram que as variantes são realmente mais difíceis (coerente com a queda
  do ROC AUC, que não depende de limiar). Mesmo recalibrados, os modelos ficam só 0,03 a 0,12 acima do chute.
- **Não dá para separar o efeito de shiny** (não sabemos quais imagens são shiny) **nem o efeito do estilo da imagem**
  (pixel art vs arte oficial) do efeito da cor.

---

## Autoria

Luciene, pós-graduação em IA.
