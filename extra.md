# Undersampling

- Usado para deixar a mesma quantidade de registros da classe majoritária e minoritária, para que o modelo não fique enviesado para a classe majoritária.

- Porém, ao fazer isso, perdemos informações importantes da classe majoritária, o que pode impactar negativamente a performance do modelo.

### Exemplo de como fazer o undersampling:

```python
fraudes = df[df['Class'] == 1]

normais = df[df['Class'] == 0].sample(
    n=len(fraudes),
    random_state=42
)

df_under = pd.concat([fraudes, normais])
```

# Oversampling
- Usado para aumentar a quantidade de registros da classe minoritária, para que o modelo não fique enviesado para a classe majoritária.
  
- Porém, ao fazer isso, podemos estar criando registros artificiais da classe minoritária, o que pode impactar negativamente a performance do modelo.
  
### Exemplo de como fazer o oversampling:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_res, y_res = smote.fit_resample(x_train, y_train)
Em fim
```

Para saber qual a melhor técnica de balanceamento, é necessário testar cada uma delas e comparar os resultados obtidos.