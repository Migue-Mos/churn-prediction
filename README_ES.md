# Predicción de abandono de clientes (Telco Customer Churn)

Proyecto de clasificación binaria para predecir qué clientes de una compañía de telecomunicaciones tienen probabilidad de cancelar su contrato (churn), construido con scikit-learn.

El foco está en un **flujo de trabajo reproducible y sin fuga de datos**: el preprocesado vive dentro de un `Pipeline` de scikit-learn, los modelos se comparan con validación cruzada solo sobre el conjunto de entrenamiento y el conjunto de test se usa una única vez, con el modelo final.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (dataset de ejemplo de IBM): 7.043 clientes y 21 columnas (datos demográficos, servicios contratados, información de la cuenta y la etiqueta `Churn`). Alrededor del 26 % de los clientes abandona, por lo que las clases están desbalanceadas.

Limpieza aplicada (ver `src/preprocessing.py`):

- Se elimina `customerID`: es un identificador único sin valor predictivo, y codificarlo con one-hot añadiría unas 7.000 columnas de ruido.
- `TotalCharges` se convierte a numérico. Se eliminan once filas con valores en blanco (clientes con `tenure = 0`), quedando 7.032 filas.
- `Churn` se codifica como 0/1.

## Estructura del proyecto

```
.
├── data/                  # aquí va Telco-Customer-Churn.csv (no se versiona)
├── models/                # pipeline guardado (no se versiona)
├── notebooks/
│   ├── 01_EDA.ipynb       # análisis exploratorio y limpieza
│   └── 02_modeling.ipynb  # comparación de modelos y evaluación final
├── src/
│   ├── data_loader.py     # carga e inspección de datos
│   ├── preprocessing.py   # limpieza y preprocesador de columnas
│   ├── modeling.py        # pipelines de los modelos y guardado
│   └── evaluation.py      # validación cruzada, curvas ROC y métricas en test
├── main.py                # script de extremo a extremo
└── requirements.txt
```

## Metodología

1. **Primero se divide.** División estratificada 80/20 en train/test (`random_state=42`) antes de ajustar nada.
2. **Preprocesado dentro del pipeline.** Un `ColumnTransformer` escala las columnas numéricas (`StandardScaler`) y aplica one-hot a las categóricas (`OneHotEncoder`). Al formar parte de cada `Pipeline`, se ajusta solo con datos de entrenamiento, y de nuevo dentro de cada fold de la validación cruzada.
3. **Selección de modelo con validación cruzada.** Se comparan cuatro candidatos (regresión logística, random forest, SVM y XGBoost) con validación cruzada estratificada de 5 folds sobre el conjunto de entrenamiento. El desbalance de clases se trata con `class_weight="balanced"` (`scale_pos_weight` en XGBoost) para que los modelos sean comparables.
4. **Evaluación final una sola vez.** El modelo elegido se entrena con todo el conjunto de entrenamiento y se evalúa una única vez sobre el test. El archivo guardado contiene el pipeline completo, preprocesado incluido.

## Resultados

### Validación cruzada sobre el conjunto de entrenamiento (5 folds, media)

| Modelo | Accuracy | Recall (churn) | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Regresión logística | 0.752 | 0.802 | 0.632 | 0.846 | 0.659 |
| Random forest | 0.769 | 0.765 | 0.637 | 0.847 | 0.663 |
| SVM | 0.748 | 0.785 | 0.623 | 0.826 | 0.592 |
| XGBoost | 0.758 | 0.791 | 0.634 | 0.848 | 0.668 |

Las desviaciones típicas entre folds rondan 0.005 en el ROC-AUC, así que la regresión logística, el random forest y XGBoost están técnicamente empatados; solo el SVM queda claramente por detrás (PR-AUC). La accuracy no se usa para elegir entre modelos: depende del umbral de decisión, que `class_weight` desplaza.

### Modelo final: regresión logística

Se eligió la regresión logística porque iguala a los mejores modelos en ROC-AUC y PR-AUC, tiene el mayor recall y es la opción más simple e interpretable. XGBoost no la mejoró.

| Métrica (test) | Valor |
|---|---|
| Accuracy | TODO |
| Recall (churn) | TODO |
| ROC-AUC | TODO |

<!-- Rellenar con la salida de `python main.py` -->

## Por qué el recall

El objetivo de un modelo de churn es señalar a los clientes que están a punto de irse para que la empresa pueda actuar (una llamada, una oferta de retención). No detectar a un cliente que se va (falso negativo) supone perder sus ingresos futuros; marcar a uno que se habría quedado (falso positivo) cuesta una oferta innecesaria. Bajo la hipótesis de que perder un cliente cuesta más que una acción de retención, se prioriza el recall de la clase churn. El dataset no incluye información de costes, por lo que esto es una suposición y no un dato medido. El equilibrio se puede ajustar más adelante moviendo el umbral de decisión.

## Cómo ejecutarlo

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Descarga el dataset de Kaggle y guárdalo como `data/Telco-Customer-Churn.csv`.
2. Ejecuta el flujo completo:

```bash
python main.py
```

Esto imprime la tabla de validación cruzada, dibuja las curvas ROC fuera de fold, evalúa el modelo final en el test y lo guarda en `models/best_churn_model.pkl`.

Para usar el modelo guardado:

```python
import joblib

model = joblib.load("models/best_churn_model.pkl")
model.predict_proba(new_customers)[:, 1]   # probabilidad de churn
```

`new_customers` es un DataFrame con las columnas originales (sin `customerID` ni `Churn`); el pipeline aplica el preprocesado por sí mismo.

## Posibles siguientes pasos

- Elegir el umbral de decisión a partir de predicciones de validación cruzada y una hipótesis de costes.
- Ajuste de hiperparámetros con `RandomizedSearchCV` sobre el conjunto de entrenamiento.
- Explicabilidad del modelo (coeficientes de la regresión logística, SHAP para modelos de árboles).
