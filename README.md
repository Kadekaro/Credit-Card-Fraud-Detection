# Credit Card Fraud Detection

Projeto de Machine Learning desenvolvido para estudar a detecção de transações fraudulentas em cartões de crédito, utilizando classificação supervisionada e comparando diferentes algoritmos.

O principal desafio do projeto é o forte desbalanceamento entre as classes. Como as fraudes representam uma parcela muito pequena das transações, o **recall** foi definido como a principal métrica de avaliação, buscando identificar o maior número possível de fraudes.

> **Nota metodológica:** os resultados apresentados neste projeto são experimentais. Durante o estudo, diferentes thresholds foram explorados utilizando o conjunto de teste. Em uma aplicação mais rigorosa, modelo e threshold deveriam ser selecionados com um conjunto de validação separado, deixando o teste exclusivamente para a avaliação final.

---

## 1. Objetivo

Desenvolver e comparar modelos de classificação capazes de identificar transações fraudulentas e analisar como fatores como pré-processamento, escolha do algoritmo e threshold de classificação influenciam o resultado.

O projeto também busca compreender, na prática:

- como lidar com um dataset altamente desbalanceado;
- quando utilizar padronização das variáveis;
- como interpretar uma matriz de confusão;
- como recall, precision e F1-score se comportam em um problema de fraude;
- como a alteração do threshold modifica as previsões;
- como comparar modelos com características diferentes.

---

## 2. Dataset

Foi utilizado o dataset **Credit Card Fraud Detection**, disponibilizado pelo Kaggle.

O conjunto possui:

| Característica | Valor |
|---|---:|
| Transações | 284.807 |
| Variáveis de entrada | 30 |
| Colunas totais | 31 |
| Transações normais (`Class = 0`) | 284.315 |
| Transações fraudulentas (`Class = 1`) | 492 |
| Normais | 99,827% |
| Fraudes | 0,173% |

As variáveis estão organizadas em:

- `Time`: tempo decorrido em segundos desde a primeira transação do dataset;
- `V1` até `V28`: características transformadas e anonimizadas;
- `Amount`: valor da transação;
- `Class`: variável target, em que `0` representa uma transação normal e `1` uma fraude.

As variáveis `V1`–`V28` são componentes transformados/anônimos. Por isso, não é possível atribuir a elas uma interpretação direta como, por exemplo, tipo de estabelecimento ou localização.

---

## 3. Análise exploratória

A análise inicial verificou a estrutura do dataset, quantidade de registros, tipos das variáveis, valores ausentes e distribuição da variável target.

Não foram encontrados valores ausentes nas 31 colunas.

O principal ponto observado foi o forte desbalanceamento das classes:

```text
Classe 0: 284.315
Classe 1:     492
```

Esse desbalanceamento influencia diretamente a avaliação dos modelos. Um classificador poderia apresentar uma acurácia muito alta simplesmente por classificar a grande maioria das transações como normais, mesmo deixando de detectar muitas fraudes.

Por esse motivo, a acurácia não foi utilizada como principal critério de decisão.

---

## 4. Preparação dos dados

### 4.1 Separação entre características e target

A coluna `Class` foi separada como variável target (`y`), enquanto as outras 30 colunas foram utilizadas como características de entrada (`X`).

### 4.2 Treino e teste

Foi utilizada uma divisão de 80% para treinamento e 20% para teste, com `random_state=42` e estratificação:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

O resultado foi:

| Conjunto | Registros | Fraudes |
|---|---:|---:|
| Treino | 227.845 | 394 |
| Teste | 56.962 | 98 |

A estratificação foi utilizada para preservar aproximadamente a mesma proporção entre as classes nos dois conjuntos.

### 4.3 Padronização

Para os modelos sensíveis à escala das variáveis, foi utilizado `StandardScaler`.

O scaler foi ajustado somente com os dados de treinamento:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Isso evita utilizar informações estatísticas do conjunto de teste durante o ajuste do pré-processamento.

A padronização foi utilizada para:

- Logistic Regression;
- SVM.

Os modelos baseados em árvores foram treinados com os dados originais, sem necessidade de padronização:

- Random Forest;
- Gradient Boosting;
- HistGradientBoosting.

---

## 5. Métricas de avaliação

### Recall

O recall mede a proporção das fraudes reais que foram identificadas pelo modelo.

$$
Recall = \frac{TP}{TP + FN}
$$

Neste projeto, essa é a métrica que recebe maior atenção, pois os **False Negatives (FN)** representam fraudes que não foram identificadas.

### Precision

A precision mede quantas das transações classificadas como fraude realmente eram fraudulentas.

$$
Precision = \frac{TP}{TP + FP}
$$

Ela é importante porque uma quantidade excessiva de falsos positivos pode fazer com que muitas transações normais sejam tratadas como suspeitas.

### F1-score

O F1-score combina precision e recall em uma única métrica por meio da média harmônica:

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### Matriz de confusão

| Resultado | Significado |
|---|---|
| **TP** | Fraude corretamente identificada |
| **FN** | Fraude classificada como normal |
| **FP** | Transação normal classificada como fraude |
| **TN** | Transação normal corretamente identificada |

---

## 6. Threshold de classificação

Os modelos de classificação podem produzir uma probabilidade associada à classe positiva. O **threshold** determina a partir de qual valor essa probabilidade será convertida em uma previsão de fraude.

Por exemplo, com threshold `0.40`:

```python
y_pred = (y_prob >= 0.40).astype(int)
```

Valores menores tendem a classificar mais transações como fraude. Isso pode aumentar o recall, mas também pode aumentar a quantidade de falsos positivos.

Durante o projeto, foram avaliados os thresholds:

```text
0.50, 0.45, 0.40, 0.35, 0.30,
0.25, 0.20, 0.15, 0.10
```

Essa análise foi realizada para compreender o equilíbrio entre recall, precision e F1-score em cada modelo.

---

# 7. Modelos avaliados

Foram comparados cinco algoritmos:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. HistGradientBoosting
5. Support Vector Machine (SVM)

---

## 7.1 Logistic Regression

A Logistic Regression foi utilizada como **baseline**, servindo como referência inicial para comparação com os demais algoritmos.

Foi utilizada a combinação:

```text
Logistic Regression
        +
StandardScaler
        +
threshold = 0.15
```

### Resultado

| Métrica | Resultado |
|---|---:|
| Recall | 76,53% |
| Precision | 72,82% |
| F1-score | 74,63% |
| TP | 75 |
| FN | 23 |
| FP | 28 |
| TN | 56.836 |

No conjunto de teste, o modelo identificou 75 das 98 fraudes.

A alteração do threshold aumentou o recall em relação ao threshold padrão, mas com redução da precision. Isso demonstra o trade-off entre encontrar mais fraudes e evitar classificações incorretas de transações normais.

---

## 7.2 Random Forest

O Random Forest foi treinado com os dados originais, sem padronização.

Configuração utilizada:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Foram testados diferentes thresholds.

### Thresholds avaliados

| Threshold | Recall | Precision | F1-score |
|---:|---:|---:|---:|
| 0,50 | 81,63% | 94,12% | 87,43% |
| 0,45 | 82,65% | 93,10% | 87,57% |
| **0,40** | **85,71%** | **93,33%** | **89,36%** |
| 0,35 | 85,71% | 89,36% | 87,50% |
| 0,30 | 87,76% | 86,00% | 86,87% |
| 0,25 | 87,76% | 81,90% | 84,73% |
| 0,20 | 87,76% | 79,63% | 83,50% |
| 0,15 | 87,76% | 78,18% | 82,69% |
| 0,10 | **88,78%** | 71,90% | 79,45% |

Entre os thresholds avaliados, `0.40` apresentou o maior F1-score.

### Resultado utilizado como referência

```text
Threshold = 0.40
```

| Resultado | Valor |
|---|---:|
| Recall | **85,71%** |
| Precision | **93,33%** |
| F1-score | **89,36%** |
| TP | **84** |
| FN | **14** |
| FP | **6** |
| TN | **56.858** |

A matriz de confusão foi:

```text
[[56858,     6],
 [   14,    84]]
```

Nesse ponto de decisão, 84 das 98 fraudes presentes no conjunto de teste foram identificadas.

É importante observar que o threshold `0.10` apresentou recall ligeiramente maior (`88,78%`), mas com precision menor (`71,90%`). Portanto, `0.40` não deve ser interpretado como um threshold universalmente melhor: ele foi utilizado como referência porque apresentou o maior F1-score entre os thresholds testados nesse experimento.

### Importância das características

A análise da importância das características mostrou:

| Posição | Feature | Importância |
|---:|---|---:|
| 1 | V17 | 0,170325 |
| 2 | V14 | 0,136363 |
| 3 | V12 | 0,133326 |
| 4 | V10 | 0,074073 |
| 5 | V16 | 0,071792 |
| 6 | V11 | 0,045277 |
| 7 | V9 | 0,031127 |
| 8 | V4 | 0,030496 |
| 9 | V18 | 0,028156 |
| 10 | V7 | 0,024627 |

V17, V14 e V12 aparecem entre as características mais importantes para as decisões das árvores.

Essa importância representa a contribuição relativa das características para as divisões realizadas pelas árvores. Ela não significa que, por exemplo, `V17 = 17% das fraudes`, nem demonstra relação causal.

Foram também comparadas as distribuições de V17, V14 e V12 entre transações normais e fraudulentas. As distribuições mostraram diferenças entre as classes, mas também apresentaram sobreposição. Portanto, nenhuma dessas características deve ser interpretada isoladamente como uma regra direta para identificar fraude.

---

## 7.3 Gradient Boosting

O Gradient Boosting foi testado como outro modelo baseado em árvores.

O primeiro experimento apresentou desempenho inferior aos modelos que tiveram melhores resultados no projeto.

Também foi realizada uma busca limitada de hiperparâmetros devido ao custo computacional do treinamento.

O espaço utilizado foi:

```python
param_dist = {
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3]
}
```

Foram avaliadas 8 combinações com validação cruzada de 3 partes, totalizando 24 ajustes.

A busca utilizou:

```python
scoring="recall"
```

A configuração encontrada foi:

```text
n_estimators = 50
max_depth = 2
learning_rate = 0.1
```

O `best_score_` foi aproximadamente **60,90% de recall médio na validação cruzada do conjunto de treino**. Esse valor não representa o recall obtido no conjunto de teste.

Na avaliação posterior do modelo ajustado, as probabilidades ficaram fortemente concentradas em valores baixos. No threshold `0.10`, por exemplo, o resultado foi:

```text
Recall    = 7,14%
Precision = 70,00%
F1        = 12,96%
```

com:

```text
TP = 7
FN = 91
FP = 3
TN = 56.861
```

Esse comportamento mostrou que a configuração ajustada não apresentou desempenho competitivo no conjunto de teste utilizado no estudo. Por isso, não foram realizados novos ciclos extensos de otimização.

> A tabela de comparação final do notebook apresenta o experimento original de Gradient Boosting com threshold `0.30`, cujo resultado foi **Recall 25,51% | Precision 59,52% | F1 35,71%**. O experimento ajustado por `RandomizedSearchCV` foi analisado separadamente acima.

---

## 7.4 HistGradientBoosting

O HistGradientBoosting foi testado como uma variação do Gradient Boosting.

Configuração utilizada:

```python
HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)
```

### Resultado no threshold 0,50

| Métrica | Resultado |
|---|---:|
| Recall | 73,47% |
| Precision | 54,55% |
| F1-score | 62,61% |

A redução do threshold aumentou pouco o recall e produziu queda significativa na precision. No threshold `0.10`, por exemplo, o recall chegou a `75,51%`, enquanto a precision caiu para `40,88%`.

---

## 7.5 Support Vector Machine (SVM)

O SVM foi utilizado como outro algoritmo de classificação para comparação.

Como o SVM é sensível à escala das variáveis, foram utilizados os dados transformados pelo `StandardScaler`.

Foi utilizado kernel RBF:

```python
SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)
```

O treinamento apresentou custo computacional elevado devido ao tamanho do dataset.

### Resultado

Entre os thresholds avaliados, `0.20` apresentou o maior F1-score:

| Métrica | Resultado |
|---|---:|
| Recall | 78,57% |
| Precision | 93,90% |
| F1-score | 85,56% |

A alteração do threshold teve pouco efeito na maior parte dos valores testados, indicando concentração das probabilidades produzidas pelo modelo.

---

# 8. Comparação final dos modelos

A tabela abaixo reproduz a comparação apresentada no notebook, utilizando os thresholds selecionados para cada experimento:

| Modelo | Threshold | Recall | Precision | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0,15 | 76,53% | 72,82% | 74,63% |
| **Random Forest** | **0,40** | **85,71%** | **93,33%** | **89,36%** |
| Gradient Boosting | 0,30 | 25,51% | 59,52% | 35,71% |
| HistGradientBoosting | 0,50 | 73,47% | 54,55% | 62,61% |
| SVM | 0,20 | 78,57% | 93,90% | 85,56% |

A comparação mostra diferentes comportamentos entre os modelos. O Random Forest, no experimento selecionado como referência, apresentou a combinação de **85,71% de recall, 93,33% de precision e 89,36% de F1-score**.

O SVM apresentou precision semelhante e recall inferior ao Random Forest nos experimentos realizados. A Logistic Regression funcionou como baseline, enquanto os modelos de Gradient Boosting testados apresentaram desempenho inferior nas configurações utilizadas.

---

# 9. Modelo utilizado como referência final

Considerando o objetivo definido no início do projeto e os experimentos realizados, o **Random Forest com threshold 0,40** foi utilizado como referência para a análise final.

### Resultado

```text
Recall    = 85,71%
Precision = 93,33%
F1-score  = 89,36%
```

### Matriz de confusão

```text
[[56858,     6],
 [   14,    84]]
```

Isso significa:

- **84** fraudes foram identificadas corretamente;
- **14** fraudes não foram identificadas;
- **6** transações normais foram classificadas como fraude;
- **56.858** transações normais foram classificadas corretamente.

Como o conjunto de teste possui apenas 98 fraudes, cada transação fraudulenta perdida tem impacto significativo no recall.

---

# 10. Principais aprendizados

Este projeto permitiu observar na prática alguns pontos importantes de Machine Learning aplicado à classificação:

### Desbalanceamento

Em problemas com classes muito desiguais, a acurácia pode transmitir uma impressão enganosa do desempenho. É necessário observar métricas relacionadas diretamente à classe de interesse.

### Recall x Precision

A escolha da métrica depende do custo dos diferentes tipos de erro. Neste projeto, o recall recebeu atenção especial porque o objetivo é evitar deixar muitas fraudes sem identificação.

### Threshold

O threshold não é simplesmente um detalhe do modelo. Alterá-lo muda diretamente o equilíbrio entre recall e precision.

### Escalonamento

O `StandardScaler` é importante para algoritmos sensíveis à escala, como Logistic Regression e SVM. Modelos baseados em árvores não dependem da mesma forma desse processo.

### Comparação de modelos

Nenhum algoritmo deve ser avaliado apenas pelo nome ou pela complexidade. O comportamento real deve ser observado nas métricas e no contexto do problema.

### Feature importance

A importância das características pode ajudar a interpretar um modelo baseado em árvores, mas não deve ser confundida com causalidade.

---

# 11. Limitações do projeto

Este projeto foi desenvolvido como estudo e experimento de Machine Learning. Os principais pontos que devem ser considerados ao interpretar os resultados são:

1. **Threshold explorado no conjunto de teste:** os thresholds foram avaliados diretamente em `X_test`/`y_test`. Em uma metodologia mais rigorosa, essa seleção seria feita em um conjunto de validação separado.
2. **Ausência de uma separação treino/validação/teste:** o projeto utiliza treino e teste, mas a seleção de modelo e threshold não foi organizada em uma terceira partição independente.
3. **Pouca otimização de hiperparâmetros:** a busca do Gradient Boosting foi deliberadamente limitada devido ao custo computacional.
4. **Dataset anonimizado:** as variáveis `V1`–`V28` não possuem interpretação direta disponível no projeto.
5. **Ambiente experimental:** os resultados não representam garantia de desempenho em um sistema bancário real.

Uma versão mais rigorosa poderia utilizar uma divisão explícita em **treino, validação e teste**, selecionar modelo e threshold na validação e utilizar o teste apenas uma vez para a avaliação final.

---

# 12. Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Kaggle / KaggleHub

---

# 13. Estrutura do projeto

Uma organização simplificada dos principais arquivos é:

```text
Credit Card Fraud Detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_analysis.ipynb
│
└── README.md
```

> A estrutura acima representa a organização lógica do projeto. Os nomes/pastas podem variar conforme a organização local do repositório.

---

# 14. Como reproduzir o projeto

O notebook foi desenvolvido em Python utilizando Jupyter Notebook/PyCharm.

De forma geral, é necessário ter as bibliotecas utilizadas instaladas e disponibilizar o arquivo `creditcard.csv` na pasta `data/` esperada pelo notebook.

A execução segue esta sequência:

```text
Carregamento dos dados
        ↓
Análise exploratória
        ↓
Separação X / y
        ↓
Train / Test Split
        ↓
Padronização quando necessária
        ↓
Logistic Regression
        ↓
Random Forest
        ↓
Gradient Boosting
        ↓
HistGradientBoosting
        ↓
SVM
        ↓
Comparação dos resultados
        ↓
Análise final do Random Forest
```

---

# 15. Conclusão

O projeto demonstrou como construir e comparar modelos de classificação para um problema de detecção de fraudes com forte desbalanceamento entre as classes.

A Logistic Regression foi utilizada como baseline, seguida por modelos baseados em árvores e por um SVM. A análise mostrou que as escolhas de algoritmo e threshold modificam significativamente o equilíbrio entre recall e precision.

Nos experimentos realizados, o **Random Forest com threshold 0,40** foi utilizado como referência final, apresentando:

- **Recall: 85,71%**
- **Precision: 93,33%**
- **F1-score: 89,36%**
- **TP: 84**
- **FN: 14**
- **FP: 6**
- **TN: 56.858**

O projeto também mostrou que um resultado de Machine Learning não deve ser analisado apenas por uma métrica isolada. A interpretação conjunta da matriz de confusão, recall, precision, F1-score e threshold é fundamental, especialmente em problemas nos quais os custos dos erros são diferentes.

Por fim, os resultados devem ser entendidos como resultados de um **estudo experimental**, e não como uma solução pronta para operação bancária. Uma evolução natural do projeto seria utilizar uma separação explícita entre treino, validação e teste para seleção do modelo e do threshold, seguida de uma avaliação final independente.
