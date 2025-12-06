# Predicción de Churn de Clientes

## Descripción del Proyecto
El objetivo de este proyecto es predecir el churn (abandono) de clientes de una compañía de telecomunicaciones utilizando Machine Learning. Esto permite identificar clientes en riesgo y aplicar acciones de retención específicas.

---

## Dataset
- Fuente: Telco Customer Churn dataset  
- Variables: información demográfica del cliente, cuenta y servicios contratados  
- Variable objetivo: `Churn` (Yes/No)

---

## Estructura del Proyecto

churn_project/
├── data/ # Dataset original y limpio para pruebas en notebook
├── src/ # Código fuente: carga, preprocesamiento, modelado, evaluación
├── notebooks/ # Análisis Exploratorio (EDA) y modelado
├── models/ # Modelos entrenados (opcional)
├── main.py # Script principal para ejecutar el pipeline
├── README.md # Este archivo

---

## Preprocesamiento de Datos
- Eliminar la columna `customerID` (no aporta información)  
- Convertir `TotalCharges` a numérico  
- Codificar `Churn` como 0/1  
- One-Hot Encoding para variables categóricas  
- Eliminar valores nulos  
- Separar Columna objetivo de categorías
- Escalar variables numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`)  

---

## Modelos Entrenados
- **Regresión Logística**: interpretable, lineal  
- **Random Forest**: robusto, maneja relaciones no lineales, mejor rendimiento como baseline 
- **SVM (Support Vector Machine)**: preciso con datos limpios y escalados  

---

## Métricas de Evaluación
- Accuracy  
- Reporte de Clasificación (Precision, Recall, F1-score)  
- Matriz de Confusión  
- Curva ROC (si el modelo permite probabilidades)  

---

## Conclusiones Clave
- **Random Forest** es el modelo más robusto.  
- Variables más asociadas al churn:
  - Tenure (meses con la compañía)  
  - Tipo de contrato (los contratos mensuales presentan más churn)  
  - Servicios contratados (la falta de servicios adicionales aumenta el churn)  
  - MonthlyCharges  
- Los pasos de limpieza (eliminar ID, arreglar TotalCharges, codificar Churn) fueron críticos.  

---

## Recomendaciones para el Negocio
- Focalizar retención en clientes nuevos (tenure bajo)  
- Incentivar contratos a largo plazo  
- Ofrecer bundles de servicios para aumentar la fidelidad  
- Revisar precios altos y ofrecer incentivos personalizados  

---

## Próximos Pasos
- Optimización de hiperparámetros (GridSearch o Bayesian)  
- Ingeniería de features (ratios, interacciones)  
- Validación cruzada para métricas robustas  
- Desplegar modelo en API o dashboard para monitorización en tiempo real  

---

## Cómo Ejecutar
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo
python main.py