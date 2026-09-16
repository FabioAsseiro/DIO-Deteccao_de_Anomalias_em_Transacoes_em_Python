import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

# # Ver proporção das classes
# print("Distribuição das classes:")
# print(df['Class'].value_counts(normalize=True))

# Transformação do Amount
df['Amount_log'] = np.log1p(df['Amount'])

# Padronização
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount_log']])

# Separação X e Y
x = df.drop('Class', axis=1)
y = df['Class']

# Separação treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

y_probs = model.predict_proba(x_test)[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_probs)

# plt.plot(fpr,tpr)
# plt.title('Curva ROC')
# plt.xlabel('Taxa de Falsos Positivos')
# plt.ylabel('Taxa de Verdadeiros Positivos')
# plt.show()

# print("AUC-ROC:", roc_auc_score(y_test, y_probs))

# precision, recall, _ = precision_recall_curve(y_test, y_probs)
# plt.plot(recall, precision)
# plt.title('Curva Precision-Recall')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.show()

# rf = RandomForestClassifier(
#     n_estimators=50,
#     max_depth=10,
#     class_weight='balanced',
#     n_jobs=-1,
#     random_state=42
# )

# rf.fit(x_train, y_train)
# y_pred_rf = rf.predict(x_test)

# print("Relatório de Classificação para Random Forest:")
# print(classification_report(y_test, y_pred_rf))

# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('model', LogisticRegression(max_iter=1000))
# ])

# pipeline.fit(x_train, y_train)

# y_pred = pipeline.predict(x_test)

# threshold = 0.3

# y_pred_custom = (y_probs > threshold ).astype(int)

# print("Relatório de Classificação com limiar personalizado (0.3):")
# print(classification_report(y_test, y_pred_custom))

xgb = XGBClassifier(
    scale_pos_weight=10, #ajuda com desbalanceamento
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb.fit(x_train, y_train)

# # y_pred_xgb = xgb.predict(x_test)

# # print("Relatório de Classificação para XGBoost:")
# # print(classification_report(y_test, y_pred_xgb))

# importancias = xgb.feature_importances_

# plt.bar(range(len(importancias)), importancias)
# plt.title('Importância das variáveis - XGBoost')
# plt.show()

# param_grid = {
#     'max_depth': [3, 5],
#     'n_estimators': [50, 100],
# }
# grid = GridSearchCV(
#     XGBClassifier(eval_metric='logloss'),
#     param_grid,
#     scoring='recall',
#     cv=3,
# )

# grid.fit(x_train, y_train)

# print("Melhores parâmetros encontrados:", grid.best_params_)

explainer = shap.Explainer(xgb)
shap_values = explainer(x_test[100])

shap.plots.bar(shap_values)

# # Resultados
# print("Relatório de Classificação:")
# print(classification_report(y_test, y_pred))