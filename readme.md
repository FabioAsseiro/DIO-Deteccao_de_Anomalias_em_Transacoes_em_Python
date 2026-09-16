# Detecção de Anomalias em Transações com Python

Projeto desenvolvido durante o bootcamp da **Digital Innovation One (DIO)** com foco em **Análise de Dados e Machine Learning**, utilizando Python para explorar e identificar possíveis transações fraudulentas em uma base de dados de cartões de crédito.

O projeto aborda etapas importantes de um fluxo de Machine Learning, desde o tratamento e preparação dos dados até a construção, avaliação e interpretação de modelos de classificação.

---

## Sobre o projeto

A detecção de transações fraudulentas é um problema de classificação que apresenta um grande desafio devido ao **desbalanceamento entre as classes**.

Neste projeto, foi utilizado um dataset de transações de cartão de crédito para explorar técnicas de preparação dos dados e modelos capazes de identificar padrões associados a possíveis fraudes.

Entre as etapas desenvolvidas estão:

* Importação e exploração dos dados
* Transformação da variável `Amount`
* Padronização dos dados
* Separação entre variáveis preditoras e variável alvo
* Divisão dos dados em treino e teste
* Treinamento de modelos de classificação
* Avaliação de modelos
* Análise de desbalanceamento das classes
* Estudo de técnicas de undersampling e oversampling
* Análise de importância das variáveis
* Interpretação do modelo utilizando SHAP

---

## Dataset

O projeto utiliza o dataset **Credit Card Fraud Detection**, disponibilizado pelo TensorFlow.

A base contém informações de transações realizadas com cartões de crédito, incluindo a variável `Class`, utilizada para identificar as transações normais e fraudulentas.

A classificação utilizada no projeto é:

| Classe | Descrição             |
| ------ | --------------------- |
| `0`    | Transação normal      |
| `1`    | Transação fraudulenta |

O dataset é carregado diretamente por meio da URL:

```python
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)
```

---

## Tecnologias utilizadas

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **SHAP**
* **Matplotlib**

### Principais conceitos praticados

* Análise exploratória de dados
* Pré-processamento
* Feature Engineering
* Normalização e padronização
* Classificação supervisionada
* Desbalanceamento de classes
* Avaliação de modelos
* Machine Learning
* Interpretabilidade de modelos

---

## Tratamento dos dados

Uma das etapas realizadas foi a transformação da variável `Amount`.

Primeiramente, foi aplicada uma transformação logarítmica utilizando `log1p`:

```python
df['Amount_log'] = np.log1p(df['Amount'])
```

Em seguida, os valores foram padronizados utilizando `StandardScaler`:

```python
scaler = StandardScaler()

df['Amount_scaled'] = scaler.fit_transform(
    df[['Amount_log']]
)
```

Essa etapa ajuda a colocar a variável em uma escala mais adequada para utilização nos modelos.

---

## Separação dos dados

As variáveis preditoras e a variável alvo foram separadas:

```python
x = df.drop('Class', axis=1)
y = df['Class']
```

Depois, os dados foram divididos entre treinamento e teste, utilizando estratificação para preservar a proporção das classes:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
```

---

## Modelos explorados

Durante o desenvolvimento foram explorados diferentes algoritmos de classificação.

### Regressão Logística

Foi utilizada como um dos primeiros modelos para estabelecer uma referência de classificação:

```python
model = LogisticRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_probs = model.predict_proba(x_test)[:, 1]
```

### Random Forest

Também foi estudada a utilização do **Random Forest**, incluindo a possibilidade de utilizar `class_weight='balanced'` para lidar com o desbalanceamento das classes.

### XGBoost

O projeto também utiliza o **XGBoost**, configurado com `scale_pos_weight` para considerar o desbalanceamento existente entre as classes:

```python
xgb = XGBClassifier(
    scale_pos_weight=10,
    eval_metric='logloss'
)

xgb.fit(x_train, y_train)
```

---

## Desbalanceamento das classes

Um dos principais desafios desse problema é o desbalanceamento entre transações normais e fraudulentas.

Por isso, foram estudadas técnicas para equilibrar as classes.

### Undersampling

O undersampling reduz a quantidade de registros da classe majoritária.

```python
fraudes = df[df['Class'] == 1]

normais = df[df['Class'] == 0].sample(
    n=len(fraudes),
    random_state=42
)

df_under = pd.concat([fraudes, normais])
```

A principal desvantagem é a possibilidade de perder informações relevantes da classe majoritária.

### Oversampling

O oversampling busca aumentar a representação da classe minoritária.

Uma das técnicas estudadas foi o **SMOTE**:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_res, y_res = smote.fit_resample(
    x_train,
    y_train
)
```

A escolha da técnica de balanceamento deve ser feita a partir da comparação dos resultados obtidos pelos modelos.

---

## Avaliação dos modelos

Entre as métricas e ferramentas exploradas estão:

* Precision
* Recall
* F1-Score
* Classification Report
* ROC-AUC
* Curva ROC
* Curva Precision-Recall

Em problemas de detecção de fraude, métricas como **Recall e Precision** são especialmente importantes, pois um modelo pode apresentar uma boa acurácia geral e ainda assim apresentar dificuldades para identificar corretamente a classe minoritária.

---

## Interpretabilidade com SHAP

Para compreender melhor o comportamento do modelo, foi utilizada a biblioteca **SHAP (SHapley Additive exPlanations)**.

```python
explainer = shap.Explainer(xgb)

shap_values = explainer(x_test[100])

shap.plots.bar(shap_values)
```

A utilização do SHAP permite analisar como as variáveis contribuem para as previsões realizadas pelo modelo.

---

## Estrutura do projeto

```text
DIO-Detec--o-de-Anomalias-em-Transa--es-em-Python/
│
├── python.py
├── extra.md
└── .gitignore
```

### `python.py`

Arquivo principal contendo o desenvolvimento do projeto, incluindo:

* Carregamento dos dados
* Pré-processamento
* Separação dos dados
* Treinamento dos modelos
* Configuração do XGBoost
* Análise com SHAP

### `extra.md`

Material complementar com anotações sobre técnicas de balanceamento de classes, incluindo:

* Undersampling
* Oversampling
* SMOTE

---

## Como executar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/FabioAsseiro/DIO-Detec--o-de-Anomalias-em-Transa--es-em-Python.git
```

### 2. Acesse a pasta

```bash
cd DIO-Detec--o-de-Anomalias-em-Transa--es-em-Python
```

### 3. Crie um ambiente virtual

```bash
python -m venv venv
```

### 4. Ative o ambiente virtual

No Windows:

```bash
venv\Scripts\activate
```

### 5. Instale as dependências

```bash
pip install pandas numpy scikit-learn matplotlib shap xgboost
```

Para trabalhar também com SMOTE:

```bash
pip install imbalanced-learn
```

### 6. Execute o projeto

```bash
python python.py
```

---

## Aprendizados

Este projeto contribuiu para a prática de conceitos importantes relacionados à área de Dados, principalmente:

* Manipulação de dados com Pandas
* Transformação e preparação de datasets
* Feature Engineering
* Machine Learning com Scikit-learn
* Classificação de dados
* Tratamento de classes desbalanceadas
* Avaliação de modelos
* XGBoost
* Interpretabilidade de modelos com SHAP
* Python aplicado à análise de dados

---

## Próximos passos

Como evolução do projeto, algumas possibilidades são:

* Comparar sistematicamente os diferentes modelos;
* Aplicar técnicas de undersampling e oversampling diretamente no treinamento;
* Testar diferentes hiperparâmetros;
* Utilizar `GridSearchCV` para otimização dos modelos;
* Comparar Precision, Recall, F1-Score e ROC-AUC;
* Ajustar o threshold de classificação;
* Criar visualizações mais completas dos resultados;
* Desenvolver um dashboard para análise das transações;
* Documentar os resultados obtidos por cada modelo.

---

## Autor

**Fábio Nogueira Asseiro**

Desenvolvedor Front-end em transição para a área de **Dados**, com foco em **Python, SQL e Análise de Dados**.

[![GitHub](https://img.shields.io/badge/GitHub-FabioAsseiro-181717?style=for-the-badge\&logo=github)](https://github.com/FabioAsseiro)

---

## Formação

Projeto desenvolvido como parte dos estudos e desafios da **Bootcamp Bradesco - GenAI, Dados & Cyber(DIO)**.

---

⭐ Se este projeto foi útil para você, considere deixar uma estrela no repositório.
