# Competencia 01 — reproducción del proceso

Este entregable contiene el feature engineering, el workflow de LightGBM, el historial de Optuna y los **12 modelos del ensamble final**. Hay dos recorridos:

- **Generar las predicciones de agosto:** cargar los modelos incluidos y promediar sus scores. No requiere Optuna, SSS ni reentrenamiento.
- **Reconstruir el proceso:** preparar los datos, recuperar o ejecutar Optuna, realizar los backtests y entrenar nuevamente el ensamble.

Los datos de entrada no están incluidos en esta carpeta. Para reproducir las predicciones se necesita el Parquet de agosto con el mismo feature engineering utilizado en el entrenamiento.

## 1. Archivos incluidos

```text
competencia01/
├── README.md
├── FE/
│   └── competencia_01_Feature_Engineering_DuckDB_v3.ipynb
├── codigos/
│   └── LGBM_Workflow_v1_4.ipynb
├── db/
│   └── optimization_lgbm_VM_V1_4.db
└── modelos/
    ├── v1_4_ensamble_t78_25_202104_202105_202106.json
    └── ...txt  (12 modelos: trials 78 y 25 × 6 semillas)
```

El manifiesto JSON contiene 1.709 nombres de variables en orden, los meses de entrenamiento, los parámetros finales y el peso de cada modelo. Cada `.txt` permite predecir sin reconstruir el entrenamiento.

## 2. Entorno y dependencias

Preparar un entorno Python con Jupyter y las dependencias de ambas notebooks. Para recrear el entorno de FE se puede usar Python 3.11: la salida guardada utiliza Python 3.11.13 y DuckDB 1.3.2. 

El FE procesa datos en DuckDB con `MEMORIA='1GB'`, `HILOS=1` y hasta `20GB` de temporales. La salida histórica guardada informa aproximadamente 7,29 GB de base y 7,43 GB de CSV. 

## 3. Datos y rutas

| Archivo necesario | Uso | ¿Incluido? |
|---|---|---|
| `competencia_01.csv` | Datos originales completos para generar variables históricas | No |
| `competencia_01_fe_v3.parquet` | Fotos etiquetadas de abril, mayo y junio de 2021 | No |
| `competencia_01_fe_v3_Ago.parquet` | Foto de agosto de 2021 con las mismas variables | No |
| Base de Optuna y modelos finales | Recuperar experimentos y predecir | Sí |

Obtener los datos de la fuente de la competencia. Para reconstruir el FE, conservar toda la historia disponible en el CSV, incluidos los meses intermedios: **filtrar abril–junio y agosto después de calcular las variables históricas**, no antes.

La notebook de modelado conserva las rutas de la VM:

```python
base_path = '/home/ds/buckets/b1'
dataset_path = base_path + '/datasets/'
modelos_path = base_path + '/V1/modelos/'
db_path = base_path + '/V1/db/'
```

En esa VM, ubicar ambos Parquet en `/home/ds/buckets/b1/datasets/`. En otro entorno, editar esas cuatro asignaciones antes de leer datos.

### Diferencias del entregable que deben considerarse
 **Rutas de modelos en el JSON:** apuntan a la VM y a nombres que tenían fecha y hora. Los `.txt` entregados fueron renombrados.


## 4. Recorrido rápido: predecir agosto con los modelos entregados

Ejecutar este bloque en una notebook con el entorno anterior. Solo requiere el manifiesto, los doce `.txt` y el Parquet de agosto. La carga desde archivo está soportada por [`lightgbm.Booster`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html).

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path('/ruta/absoluta/competencia01')  # Editar.
DATASETS = Path('/home/ds/buckets/b1/datasets')  # Editar si corresponde.

archivo_manifiesto = ROOT / 'modelos' / 'v1_4_ensamble_t78_25_202104_202105_202106.json'
manifiesto = json.loads(archivo_manifiesto.read_text(encoding='utf-8'))
nombre_ensamble = archivo_manifiesto.stem

datos_agosto = pd.read_parquet(DATASETS / 'competencia_01_fe_v3_Ago.parquet')
datos_agosto = datos_agosto.loc[datos_agosto['foto_mes'] == 202108]
X_agosto = datos_agosto.loc[:, manifiesto['columnas']]
proba_agosto = np.zeros(len(X_agosto), dtype=np.float64)

for registro in manifiesto['modelos']:
    archivo = next((ROOT / 'modelos').glob(
        f"*trial{registro['trial']}_seed{registro['seed']}.txt"
    ))
    modelo = lgb.Booster(model_file=str(archivo))
    proba_agosto += registro['peso_ensamble'] * modelo.predict(
        X_agosto, num_iteration=registro['arboles_entrenados']
    )
    print(f"Trial {registro['trial']} | Semilla {registro['seed']}: predicción completa")
    del modelo

ranking_agosto = pd.DataFrame({
    'numero_de_cliente': datos_agosto['numero_de_cliente'].to_numpy(),
    'probabilidad': proba_agosto,
}).sort_values('probabilidad', ascending=False, kind='stable').reset_index(drop=True)
```

Mantener solamente los doce modelos entregados en esa carpeta, o adaptar la selección de archivos si se agregan otras ejecuciones con el mismo trial y semilla. Seleccionar las columnas del manifiesto conserva su orden y excluye las variables `ccajas_depositos`, sin necesidad de regenerar targets para agosto.


## 5. Reconstrucción del feature engineering

Abrir [la notebook de FE](FE/competencia_01_Feature_Engineering_DuckDB_v3.ipynb) y ejecutar las celdas en orden, con `CREAR_FLAGS=False`. La configuración original busca `Datasets/competencia_01.csv` en el directorio actual o sus padres, y escribe en `Clases/Clase 4/FE`.

Para trabajar dentro del entregable, crear `Datasets/`, colocar allí el CSV original y reemplazar las asignaciones de búsqueda de raíz y carpeta de salida de la primera celda de código por:

```python
RAIZ = Path('/ruta/absoluta/competencia01')  # Editar.
CARPETA_FE = RAIZ / 'FE'
CSV_ORIGEN = RAIZ / 'Datasets' / 'competencia_01.csv'
```

Mantener las asignaciones siguientes de `BASE`, `CSV_FINAL` y `TEMPORALES`, que dependen de `CARPETA_FE`. La notebook crea reglas de negocio, ratios, lags de uno y dos meses, deltas y agregados sobre observaciones anteriores del cliente. Los acumulados excluyen la foto actual y propagan faltantes según las fórmulas de la notebook. No activar los indicadores `*_es_nulo` para reproducir esta versión.

`REHACER=False` permite recuperar etapas terminadas en DuckDB. Cambiarlo a `True` solo para recalcular una base existente cuando cambian la receta o los datos. Consumir la tabla final después de completar las validaciones y obtener la marca `v3_completa`.

Las salidas históricas guardadas en la notebook registran 983.061 filas, 1.718 columnas y 26 controles aprobados. 

### Puente necesario: generar los dos Parquet

La notebook incluida exporta DuckDB y CSV; **no contiene una exportación activa a Parquet**. 

## 6. Recuperar Optuna y ejecutar el workflow

Abrir [LGBM_Workflow_v1_4.ipynb](codigos/LGBM_Workflow_v1_4.ipynb). Ejecutar los imports, la configuración de rutas, la carga y las transformaciones iniciales. 


### Reutilizar el estudio entregado

Para reproducir la selección histórica, reemplazar la creación de `study` por la carga explícita del estudio ya creado. 

La base contiene tres estudios. El estudio `_full_4` tiene **81 trials completos y 3 fallidos** en la copia entregada; allí existen los trials 78 y 25. Su mejor trial es el 78, con valor de Optuna de $435.765.000.


## 7. Orden de revalidación y entrenamiento final

| Paso de la notebook | Configuración activa | Salida principal |
|---|---|---|
| 2. Preparación | Top 10 de Optuna; SSS estratificado, **5 repeticiones y 50 % para entrenamiento** | Funciones y `trials_por_numero` |
| 2.A | Muestra del 50 % de abril → mayo completo | `df_sss_A`, `resumen_A` |
| 2.B | Muestra del 50 % de mayo → junio completo | `df_sss_B`, `resumen_B` |
| Comparación | Prioridad de A | `comparacion_temporal` y boxplots |
| 3. Candidatos | Cinco mejores de A, lista editable | `trials_evaluar` |
| 3. Predicciones | Abril + mayo completos → junio; un entrenamiento por candidato | Predicciones de junio |
| 3. Pseudo público/privado | **1.000 ShuffleSplit sin estratificación**, 25 % público / 75 % privado | Tablas y regret |
| 4. Configuración final | Trials 78 y 25, ajustes manuales | `configuraciones_finales` |
| 5. Ensamble | Abril + mayo + junio completos; 12 modelos | Modelos, manifiesto y ranking de agosto |
| 6. Exportación | Cortes manuales | CSV de IDs |


El paso 3 entrena una vez por candidato. Para cambiar la cantidad de particiones, modificar `n_splits_pp` y repetir solo la celda de particionado y resúmenes. En el código entregado, el print por partición está comentado. Para reproducir la comparación concreta discutida durante el desarrollo, se puede fijar la lista comentada `[55, 78, 53, 25, 80]`; para seguir la selección automática actual, mantener `resumen_A.head(5)`.

La curva de referencia del paso 4 requiere que el trial 78 haya sido incluido en `trials_evaluar`, porque reutiliza sus predicciones de junio. Es una curva de un modelo entrenado con abril + mayo, no del ensamble definitivo.

### Ensamble incluido

| Parámetro | Trial 78 | Trial 25 |
|---|---:|---:|
| Rondas de Optuna | 3.194 | 2.780 |
| Rondas finales: factor 1,1 y redondeo | 3.513 | 3.058 |
| `num_leaves` | 65 | 47 |
| `min_sum_hessian_in_leaf` final: factor 1,5 | 210,313113 | 223,427096 |
| `min_gain_to_split` final: factor 1,5 | 0,708610 | 0,984553 |
| `learning_rate` | 0,005 | 0,005 |

Semillas para cada configuración: **1977, 1978, 2000, 2001, 2003 y 2007**. Cada uno de los doce modelos pesa `1/12`: se promedian scores, se ordenan de mayor a menor y se eligen los primeros N clientes. No se votan clases ni se promedian ganancias.

Para **solo reentrenar el ensamble**: ejecutar la preparación de datos, cargar el estudio, ejecutar la celda de preparación del paso 2 que define `trials_por_numero` y `columnas_modelo`, luego la configuración del paso 4 y el entrenamiento del paso 5. Omitir la curva de referencia, que sí depende de las predicciones del paso 3.

El bloque final guarda cada `.txt` después de entrenarlo y escribe el JSON al finalizar todos los modelos. No implementa reanudación automática: después de una interrupción, los modelos guardados pueden cargarse para predecir, pero repetir el bucle original vuelve a entrenar.

