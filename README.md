En este repositorio, encontraremos la práctica de NLP donde se muestran dos modelos de Machine Learning a través de los cuales se hace un análisis de sentimiento. 
Dado que los archivos necesarios tienen un peso muy grande para github, los archivos .ipynb han sido ejecutados, pero los archivos de los modelos han sido eliminados para conseguir subirlos sin exceder el límite de 100 MB por repositorio impuesto por github. 

Estructura de archivos:
--------------------------------------------------------------------------------------------------------
Como vemos, en la carpeta raíz tenemos los 4 notebooks necesarios para la práctica, y 2 versiones del notebook 3 (División y entrenamiento). En uno, encontramos los modelos de ML creados con un dataset sobre 5 puntos (el 3.0) mientras que en el otro, encontramos que el archivo se hace con un corpus entrenado de manera binaria (sólo dos posibles notas, positivo (1) y negativo (0) )

Además, he incluido un archivo .py en el que he ido copiando las funciones de preprocesamiento, de manera que puedan ser importadas tantas veces como quieran (en caso de tener que utilizarlas en más de un notebook)



├── 1.Exploración_de_datos.ipynb
├── 2.Preprocesado.ipynb
├── 3.0.Division_y_entrenamiento.ipynb
├── 3.1.division_y_entrenamiento0-1.ipynb
├── 4.métricas_y_conclusiones.ipynb
├── dataset
│   └── reviews_Video_Games_5.json.gz
├── funciones_preprocesamiento.py
├── logistic_regression_model.pkl
├── modelos
│   ├── DistilBERT_lr_0-1
│   │   ├── classification_report.json
│   │   ├── config.json
│   │   ├── confusion_matrix.csv
│   │   ├── distilbert
│   │   │   ├── config.json
│   │   │   └── model.safetensors
│   │   ├── logreg_model.joblib
│   │   ├── splits.json
│   │   ├── test_predictions.csv
│   │   └── tokenizer
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   ├── DistilBERT_lr_0-5
│   ├── Logistic_regression_bag_of_words0-1
│   │   ├── model.pkl
│   │   ├── test_data.pkl
│   │   └── vectorizer.pkl
│   ├── Logistic_regression_bag_of_words0-5
│   │   ├── model.pkl
│   │   ├── test_data_raw.pkl
│   │   └── vectorizer.pkl
│   ├── Logistic_regression_tf_idf0-1
│   │   ├── model.pkl
│   │   ├── test_data.pkl
│   │   ├── test_data_raw.pkl
│   │   └── vectorizer.pkl
│   └── Logistic_regression_tf_idf0-5
│       ├── model.pkl
│       ├── test_data.pkl
│       └── vectorizer.pkl
├── reviews_Video_Games_5_balanced.csv
├── reviews_Video_Games_5_balanced_preprocessed.csv
├── reviews_Video_Games_5_balanced_preprocessed_0-1.csv
└── __pycache__



    └── funciones_preprocesamiento.cpython-312.pyc
