# Creamos una función para cargar modelo, vectorizador y datos de test guardados

def load_model_artifacts(model_dir):
    """
    Carga el modelo, el vectorizador y los datos de test desde el directorio especificado.
    Parámetros:
    - model_dir: Ruta al directorio donde se encuentran los artefactos guardados.
    Retorna:
    - model: El modelo cargado.
    - vectorizer: El vectorizador cargado.
    - X_test_text: Los datos de test en formato de texto.
    - y_test: Las etiquetas de test.
    
    """
    import joblib
    import os

    # Cargar modelo
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    # Cargar vectorizador
    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
    # Cargar datos de test 
    X_test_text, y_test = joblib.load(os.path.join(model_dir, "test_data.pkl"))

    return model, vectorizer, X_test_text, y_test

def plot_evaluation_metrics_binary(y_test, y_pred):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.metrics import precision_recall_fscore_support

    # obtenemos precisión, recall y f1 por clase
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )

    classes = sorted(set(y_test))
    x = range(len(classes))

    # --- GRÁFICA PRECISIÓN / RECALL / F1 ---
    plt.figure(figsize=(8, 5))
    plt.bar(x, precision, width=0.25, label='Precisión')
    plt.bar([i + 0.25 for i in x], recall, width=0.25, label='Recall')
    plt.bar([i + 0.50 for i in x], f1_score, width=0.25, label='F1-Score')

    plt.xticks([i + 0.25 for i in x], classes)
    plt.title("Métricas de evaluación (Binario)")
    plt.legend()
    plt.show()

    # --- MATRIZ DE CONFUSIÓN ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusión (Binario)")
    plt.show()

    # creamos una función para graficar las métricas de evaluación
def plot_metrics (y_test, y_pred):
    unique_classes = len(set(y_test))
    if unique_classes == 2:
        print ('clasificación binaria')
        plot_evaluation_metrics_binary (y_test, y_pred)
    else:
        print ('clasificación multiclase')
        plot_evaluation_metrics_multiclass (y_test, y_pred)

def plot_evaluation_metrics_multiclass(y_test, y_pred):
    """
    Grafica las métricas de evaluación del modelo.
    Parámetros:
    - y_test: Las etiquetas verdaderas.
    - y_pred: Las etiquetas predichas por el modelo.
    Retorna:
    - None: Muestra las gráficas de las métricas.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.metrics import precision_recall_fscore_support
    # Ajustamos el tamaño de y_pred para que acepte tanto modelos binarios como multiclase
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    classes = sorted(set(y_test))
    x = range(len(classes))
    # Gráfica de precisión, recall y f1-score
    plt.figure(figsize=(10, 6))
    plt.bar(x, precision, width=0.2, label='Precisión', align='center')
    plt.bar([i + 0.2 for i in x], recall, width=0.2, label='Recall', align='center')
    plt.bar([i + 0.4 for i in x], f1_score, width=0.2, label='F1-Score', align='center')
    plt.xticks([i + 0.2 for i in x], classes)
    plt.xlabel('Clases')
    plt.ylabel('Métricas')
    plt.title('Métricas de Evaluación por Clase')
    plt.legend()
    plt.show()

    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.show()

    import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Ruta al directorio del modelo guardado

#Creamos una función para mostrar evaluación del modelo cargado
def evaluate_loaded_model(model, vectorizer, X_test_text, y_test):
    """
    Evalúa el modelo cargado utilizando los datos de test proporcionados.
    Parámetros:
    - model: El modelo cargado.
    - vectorizer: El vectorizador cargado.
    - X_test_text: Los datos de test en formato de texto.
    - y_test: Las etiquetas de test.
    Retorna:
    - None: Imprime las métricas de evaluación.
    """
    # Transformar los datos de test
    X_test_vec = vectorizer.transform(X_test_text)
    # Realizar predicciones
    y_pred = model.predict(X_test_vec)
    y_test = y_test.tolist()
    # Evaluación
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    #ver matriz de confusión
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    return y_pred
    