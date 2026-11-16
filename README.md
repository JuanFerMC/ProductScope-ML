# Amazon Product Sales — ProductScope-ML
- Sebastian B. Ruiz - s.barrera@udea.edu.co
- Ana Maria G. Grisales - Anam.garzon@udea.edu.co
- Juan F. Mona Cano - juanf.mona@udea.edu.co 
---

Pipeline de limpieza y transformación del dataset de productos de Amazon, con el objetivo de
preparar los datos para modelar el **volumen de ventas recientes** (`bought_in_last_month`).

El pipeline genera **4 datasets** (`dataset_v1.csv` … `dataset_v4.csv`) que representan distintas
estrategias de imputación, para compararlas empíricamente en la fase de modelado.

---

## Estructura del repositorio

```
Amazon product sales/
├── README.md                    # Este archivo
├── Limpieza de datos/           # Pipeline de limpieza y documentación
│   ├── amazon_sales_data_uncleaned.csv   # Entrada cruda (42.675 × 16)
│   ├── limpieza_4_datasets.ipynb         # Notebook guía (pipeline completo)
│   ├── dataset_v1.csv … dataset_v4.csv   # Salidas (se regeneran)
│   ├── LIMPIEZA_DATOS.md                 # Documento técnico de decisiones
│   └── VARIABLES.md                      # Diccionario de variables
├── Modelos de ejemplo/          # Notebooks de laboratorio (referencia)
│   ├── Copy_of_lab2_parte1.ipynb
│   └── Copy_of_lab2_parte2.ipynb
└── Modelos entrenados/          # Notebooks, modelos y resultados de KNN, SVR y otros
    ├── KNN.ipynb
    ├── SVM.ipynb                 # SVR para predecir una variable continua
    └── Models_Results/           # Métricas, predicciones, búsquedas y modelos
```

| Carpeta | Contenido |
|---|---|
| **`Limpieza de datos/`** | Todo el pipeline: CSV de entrada, notebook, los 4 datasets generados y la documentación técnica (`LIMPIEZA_DATOS.md`, `VARIABLES.md`). |
| **`Modelos de ejemplo/`** | Notebooks de laboratorio usados como referencia. |
| **`Modelos entrenados/`** | Notebooks de KNN, SVR y otros modelos, más modelos serializados y sus resultados. |
| **`Informes`** | Carpeta la cual guarda los respectivos informes E1 y E2 en sus diferentes versiones |

> **Nota:** el pipeline trabaja con rutas relativas a la carpeta `Limpieza de datos/`, así que el
> notebook debe ejecutarse **desde dentro de esa carpeta** (ver más abajo).

---

## Requisitos

- **Python 3.14**
- Dependencias: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `jupyter`
- El entorno virtual ya existe en `.venv/`

Si necesitas recrearlo desde cero:

```bash
python -m venv .venv
.venv/bin/pip install numpy pandas scipy scikit-learn matplotlib jupyter
```

---

## Cómo interactuar con el proyecto

El notebook se ejecuta **desde dentro de la carpeta `Limpieza de datos/`**.

### Opción A — Ejecutar el notebook completo (regenera los 4 datasets)

```bash
cd "Limpieza de datos"
../.venv/bin/python -m jupyter nbconvert --to notebook --execute limpieza_4_datasets.ipynb \
    --ExecutePreprocessor.kernel_name=amazon-venv
```

Al terminar, se habrán (re)generado `dataset_v1.csv` … `dataset_v4.csv` dentro de
`Limpieza de datos/`.

### Opción B — Abrir el notebook de forma interactiva

```bash
cd "Limpieza de datos"
../.venv/bin/jupyter notebook limpieza_4_datasets.ipynb
```

(o `jupyter lab`). Selecciona el kernel **`amazon-venv`** y ejecuta las celdas de arriba hacia
abajo con `Shift+Enter` o *Run All*.

---

## Qué hace el pipeline (resumen)

1. **Inventario** del dataset (nulos, cardinalidad).
2. **Eliminar duplicados** exactos (962 filas).
3. **Objetivo `bought_in_last_month`**: parseo estricto del patrón `N+ bought in past month`,
   vista previa de lo que se elimina, uso del **punto medio** del intervalo.
4. **Variables excluidas**: `image_url`, `product_url`, `collected_at`, `Price_on_variant`.
5. **Transformaciones**: `Rating`, reviews, precio, banderas, `buy_box_status`, `sustainability_badges`
   (con prueba χ² + Cramér's V), `Title`, `delivery_details`.
6. **Transformación logarítmica** (`log1p`) sobre numéricas y objetivo.
7. **Generación de los 4 datasets** (diseño 2×2 precio × delivery).

El detalle de cada decisión está en **`Limpieza de datos/LIMPIEZA_DATOS.md`** y el significado
de cada variable en **`Limpieza de datos/VARIABLES.md`**.

---

## Los 4 datasets: qué representa cada uno

| Variante | Precio vacío | Delivery | Filas | Columnas |
|---|---|---|---|---|
| **v1** | Eliminar fila | Imputar costo con 0 | 24.436 | 20 |
| **v2** | Eliminar fila | Sin columna `delivery_cost` | 24.436 | 18 |
| **v3** | Imputar mediana | Imputar costo con 0 | 31.202 | 20 |
| **v4** | Imputar mediana | Sin columna `delivery_cost` | 31.202 | 18 |

- **v1 vs v3** → mide el efecto de **eliminar vs imputar** el precio.
- **v2 vs v4** → lo mismo, pero sin la columna de envío.
- **v1 vs v2** y **v3 vs v4** → mide el **aporte de `delivery_cost`**.

Las cuatro variantes comparten el mismo pipeline; solo cambian esas dos decisiones.

### Columnas de salida

**Comunes a todas (18):**
`Rating`, `Number_of_reviews`, `has_discount`, `is_best_seller_flag`, `is_sponsored_flag`,
`has_coupon`, `buy_box_status`, `is_free_delivery`, `title_word_count`, `has_sustainability_badge`,
`bought_in_last_month`, `bought_cat`, `current_price`, `is_price_outlier`, `log_price`,
`log_reviews`, `log_bought`, `log_title_words`.

**Adicionales en v1 y v3 (2):**
`delivery_cost`, `log_delivery_cost`.

> **Importante al modelar:** el objetivo es `log_bought` (= `log1p` de `bought_in_last_month`).
> Tanto `bought_in_last_month`, `bought_cat` como los demás `log_*` derivados deben **excluirse
> de las features** para evitar fuga de información.

---

## Notas para la fase de modelado

- Usa `bought_cat` solo para **estratificar** la partición, nunca como feature.
- La cola del objetivo es muy asimétrica → evalúa métricas en escala log y des-transforma con
  `expm1` si reportas en unidades.
- Las variantes v1/v2 tienen menos filas que v3/v4; al comparar modelos tenlo en cuenta
  (idealmente, evaluar sobre un conjunto de test común).
- `buy_box_status` tiene solo 2 valores efectivos (`add_to_cart` / `unknown`); conviértela a dummy.
- Cada variable numérica existe como par `X` + `log_X`: no uses ambas a la vez en modelos lineales
  (multicolinealidad); los árboles toleran cualquiera.

---

## Modelos de regresión: KNN y SVR

Los notebooks [`Modelos entrenados/KNN.ipynb`](Modelos%20entrenados/KNN.ipynb) y
[`Modelos entrenados/SVM.ipynb`](Modelos%20entrenados/SVM.ipynb) predicen
`log_bought = log1p(bought_in_last_month)` a partir de los datasets v1–v4. Como el objetivo es
continuo, el modelo de soporte vectorial es **SVR (Support Vector Regression)**, no SVC de
clasificación. Ambos notebooks mantienen las transformaciones y los datos generados por el pipeline
de limpieza; no eliminan registros por tener ventas altas.

### Cómo ejecutar los notebooks

Primero genera los datasets siguiendo las instrucciones de la sección anterior. Después, desde la
raíz del repositorio:

```bash
cd "Modelos entrenados"
../.venv/bin/jupyter notebook
```

Abre `KNN.ipynb` o `SVM.ipynb` y ejecuta sus celdas en orden. Los notebooks crean o actualizan sus
artefactos en `Modelos entrenados/Models_Results/knn/` y
`Modelos entrenados/Models_Results/svm/`, respectivamente. KNN guarda el `split_indices.json`
compartido que SVR utiliza para conservar el mismo test. Los artefactos actuales también permiten
consultar resultados sin volver a entrenar.

### Protocolo común y variables

- El objetivo modelado es `log_bought`. `bought_in_last_month` y `bought_cat` no se usan como
  predictores; `bought_cat` solo sirve para estratificar los folds.
- Las variables predictoras son indicadores de promociones y características del producto, `Rating`,
  variables transformadas como `log_price`, `log_reviews` y `log_title_words`, e
  `buy_box_unknown`. v1/v3 incluyen además `log_delivery_cost`; v2/v4 no tienen esa variable.
- Ambos modelos excluyen `is_price_outlier` como predictor siguiendo la selección de KNN. Esto no
  implica eliminar filas con ventas altas: la cola del **objetivo de ventas** se conserva.
- Las métricas primarias de selección son MAE en escala logarítmica; también se reportan RMSE y R²
  en log, además de métricas en unidades tras aplicar `expm1`.
- El test de 4.887 filas se congela antes de ajustar hiperparámetros. La partición agrupa filas que
  tienen el mismo vector completo de características seleccionadas, de modo que dichos vectores no
  aparezcan simultáneamente en entrenamiento y test. Esto es una decisión de evaluación, no una
  regla de que productos distintos no puedan compartir características.
- Los CSV ya contienen imputaciones y límites calculados durante la limpieza antes de la partición.
  Los pipelines ajustan escalado e imputación interna en los folds, pero no pueden deshacer ese
  preprocesamiento global previo; las métricas deben interpretarse teniendo en cuenta esta limitación.

### KNN de regresión

[`KNN.ipynb`](Modelos%20entrenados/KNN.ipynb) implementa KNN manualmente con distancias calculadas
mediante `cdist` y verifica el comportamiento frente a `KNeighborsRegressor`. Evalúa
`k ∈ {1, 3, 5, 7, 10, 15, 25, 40, 60, 100}`, pesos uniformes o inversamente proporcionales a la
distancia y distancias euclidiana o Manhattan (`cityblock`). El escalado se ajusta solo con el train
de cada fold. Se congela un test del 20 % mediante un split estratificado por grupos y la búsqueda
usa 4 folds agrupados sobre el 80 % restante.

La configuración seleccionada para v1 es `k=100`, `weights=distance`, `metric=cityblock`. En test
v1 obtuvo MAE log = **0,5000**, RMSE log = **0,7617**, R² log = **0,4527** y MAE = **493,3 ventas**
en unidades. Como referencia, el baseline que predice la mediana de train obtuvo MAE log = 0,8452.
En validación, las cuatro variantes seleccionaron la misma configuración:

| Variante | MAE log medio | IC 95 % del MAE | Configuración |
|---|---:|---:|---|
| v1 | 0,5212 | [0,4644; 0,5713] | k=100, distancia, Manhattan |
| v2 | 0,5260 | [0,4642; 0,5869] | k=100, distancia, Manhattan |
| v3 | 0,5252 | [0,4724; 0,5752] | k=100, distancia, Manhattan |
| v4 | 0,5300 | [0,4709; 0,5920] | k=100, distancia, Manhattan |

El notebook documenta además la distribución del objetivo, comparación con el baseline, errores por
categoría de ventas y compara protocolos de partición. Una partición aleatoria estima el desempeño
en filas de una distribución similar; una partición agrupada estima el desempeño en filas cuyo
vector completo de características no aparece en entrenamiento. Que dos filas tengan las mismas
características no demuestra que sean el mismo producto ni constituye por sí solo fuga de
información. Las métricas de los dos protocolos responden a escenarios distintos y deben
identificarse por separado.

### SVR (máquina de soporte vectorial para regresión)

[`SVM.ipynb`](Modelos%20entrenados/SVM.ipynb) compara configuraciones RBF y selecciona
hiperparámetros por variante con MAE logarítmico. La búsqueda explora `C ∈ {0,01; 0,025; 0,05;
0,075; 0,1}`, `epsilon ∈ {0,025; 0,05; 0,075}` y `gamma ∈ {0,005; 0,01; 0,015}`. Por costo
computacional, cada búsqueda usa una muestra agrupada de aproximadamente 2.000 filas y
`StratifiedGroupKFold` de 4 folds. El pipeline coloca imputación mediana, `StandardScaler` y SVR
dentro de cada fold; el modelo final se ajusta con todo el conjunto de entrenamiento.

| Variante | Mejor C | Epsilon | Gamma | MAE log test | RMSE log test | R² log test | MAE en ventas |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 0,05 | 0,025 | 0,015 | 0,4834 | 0,7482 | 0,4720 | 527,8 |
| v2 | 0,075 | 0,025 | 0,015 | 0,4836 | 0,7446 | 0,4770 | 507,7 |
| v3 | 0,01 | 0,025 | 0,015 | 0,4844 | 0,7706 | 0,4398 | 545,5 |
| v4 | 0,05 | 0,05 | 0,005 | 0,5041 | 0,7769 | 0,4306 | 530,4 |

#### Cola pesada de ventas e interpretación

El volumen de ventas tiene una cola derecha pesada: el negocio puede tener productos con pocas
ventas y otros con muchísimas. Esos valores extremos son observaciones posibles y relevantes, no
errores que deban eliminarse solo por ser extremos. La transformación logarítmica ayuda a modelar
la escala, pero no hace que SVR represente perfectamente toda la heterogeneidad de la demanda.

En el test, el grupo `>500` tiene MAE log entre **0,94 y 0,99** y MAE en unidades entre **1.903 y
2.099 ventas** según la variante; en los demás grupos, el MAE log está aproximadamente entre 0,31
y 0,39. Esto muestra una limitación predictiva de SVR en la cola alta con las características
disponibles, no un fallo de ejecución. Los resultados se reportan conservando los casos de ventas
altas.

KNN v1 y SVR v1 evalúan el mismo test común: sus MAE log son 0,5000 y 0,4834, respectivamente. La
diferencia es pequeña y no debe interpretarse como una mejora estadísticamente significativa sin
una comparación inferencial adicional. Las filas de test de v1–v4 representan los mismos productos,
por lo que las métricas permiten observar el efecto de las variantes del dataset; las métricas de
SVR por variante también reflejan cambios en los datos de entrenamiento.

### Artefactos generados

| Carpeta | Contenido principal |
|---|---|
| `Modelos entrenados/Models_Results/knn/` | `metrics.json`, `results_summary.csv`, `hyperparameter_search.csv`, `grid_agg.csv`, `test_predictions.csv`, `split_indices.json`, `knn_final.joblib` y gráficas. |
| `Modelos entrenados/Models_Results/svm/` | `metrics.json`, `validation_summary.csv`, `hyperparameter_search.csv`, `test_predictions.csv`, `error_by_sales_category.csv`, `svr_v1_final.joblib`–`svr_v4_final.joblib` y gráficas. |

Los detalles de configuración, validación, limitaciones y resultados completos están en los notebooks
y en los archivos `metrics.json` de cada carpeta.

---

## Datos versionados

Los `dataset_v*.csv` son **regenerables** ejecutando el notebook y no necesitan subirse al
repositorio. El CSV crudo (`Limpieza de datos/amazon_sales_data_uncleaned.csv`) sí se versiona
como fuente de verdad del proyecto.
