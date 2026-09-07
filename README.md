# Examen Machine Learning I
Alumna: Paulina Godoy
Docente: Danilo Gómez
Universidad Mayor

1. Declaración de metadatos del dataset
    - Nombre del dataset: California Housing Dataset
    - Fuente: StatLib
    - URL: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
    - Número de filas: 20640
    - Número de columnas: 9
    - Variable objetivo: MedHouseVal
    - Tipo de tarea: Regresión

2. Metodología resumida
    - EDA y calidad de datos: verificación de valores nulos (0% en todas las columnas), cálculo de estadísticas descriptivas.
    - Tratamiento de outliers (IQR): filtrado simultáneo por rango intercuartílico (1.5×IQR) sobre las 6 variables numéricas clave. Se eliminó un 18.40% de las observaciones (3,798 de 20,640), concentradas principalmente en AveBedrms (6.90%) y Population (5.79%).
    - Análisis de la variable objetivo: skewness = 0.9265 (bajo el umbral de 1.0, por lo que no se aplicó transformación logarítmica). Se identificó censura de datos en el valor 5.0 (500,000 USD).
    - EDA de correlaciones: identificación de MedInc (r=0.6295) y AveOccup (r=-0.3228) como predictores más relacionados con el target, y multicolinealidad crítica entre Latitude/Longitude (r=-0.9336).
    - Preprocesamiento sin data leakage: partición train/test (80/20, random_state=42) previa a cualquier transformación; ColumnTransformer con SimpleImputer (mediana) + StandardScaler, ajustado solo sobre X_train.
    - Reducción de dimensionalidad (PCA): 5 componentes principales seleccionados, explicando el 88.29% de la varianza acumulada.
    - Clustering (K-Means): K=2 seleccionado como óptimo (Silhouette=0.2471), revelando una segmentación geográfica (Bahía de San Francisco vs. Los Ángeles), no socioeconómica.
    - Modelado supervisado: entrenamiento de Lasso (modelo penalizado) y RandomForestRegressor (modelo de árboles), ambos ajustados con GridSearchCV (cv=5).
    - Evaluación: cálculo de RMSE, MAE, R² y MAPE sobre el conjunto de test; análisis de sobreajuste (comparación train vs. test); análisis de importancia de variables y de las 10 observaciones con mayor error.

3. Resultados del mejor modelo
    - Modelo seleccionado: Random Forest Regressor
    - Mejores hiperparámetros:
        - Random Forest: n_estimators=200, max_depth=None
        - Lasso: alpha=0.001
    - Variables más imporantes en Random Forest: MedInc (43.45%), AveOccup (16.82%), Longitude (11.13%), Latitude (10.68%), HouseAge (6.64%)

| Métrica | Random Forest | Lasso |
|---|---|---|
| RMSE | 0.5002 | 0.6544 |
| MAE | 0.3282 | 0.4867 |
| R² | 0.7807 | 0.6246 |
| MAPE | 18.35% | 29.92% |
| Tiempo de entrenamiento | 40.61 s | 2.32 s |

**Nota:** Se detectó sobreajuste en Random Forest (RMSE train = 0.1798 vs. RMSE test = 0.5002), atribuible a la ausencia de límite de profundidad en los árboles. Ver el notebook para el análisis completo de limitaciones.

4. Video explicativo

[Video presentación](https://drive.google.com/file/d/1NfFDVkjSG1-X3WLKpaShZA5MS9wLiJSC/view?usp=sharing)


5. Cómo reproducir el análisis
    - Requisitos previos: Python 3.9 o superior, y pip.
    - Instrucciones:
    # 1. Clonar el repositorio
    git clone https://github.com/[tu-usuario]/ML1_ExamenAplicado_Apellido_Nombre.git
    cd ML1_ExamenAplicado_Apellido_Nombre

    # 2. (Opcional) Crear y activar un entorno virtual
    python -m venv venv
    source venv/bin/activate      # En Windows: venv\Scripts\activate

    # 3. Instalar dependencias
    pip install -r requirements.txt

    # 4. Abrir y ejecutar el notebook
    jupyter notebook ML1_ExamenAplicado_California_Housing.ipynb

    Ejecutar todas las celdas en orden (Kernel → Restart & Run All) para reproducir el análisis completo, desde la carga de datos hasta las métricas finales y gráficos.

6. Estructura del repositorio
```
ML1_ExamenAplicado_Apellido_Nombre/
├── ML1_ExamenAplicado_California_Housing.ipynb   # Notebook principal ejecutado
├── README.md                                      # Este archivo
├── requirements.txt                               # Dependencias (pip freeze)
├── model_comparison_metrics.csv                   # Tabla comparativa de modelos exportada
└── figures/                                        # Gráficos generados (dpi=150)
```

7. Declaración de uso de Inteligencia Artificial

    Este trabajo utilizó asistencia de IA generativa (Claude) para depuración y mejoras en código, apoyo en formato markdown para el archivo README.md y redacción de borradores para el guión del video. Las justificaciones, interpretaciones y análisis son de autoría humana y fueron verificadas con inteligencia artificial antes de su entrega final. 