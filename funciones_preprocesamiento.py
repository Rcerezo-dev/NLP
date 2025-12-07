from collections import Counter
import matplotlib.pyplot as plt
import re
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import pandas as pd
stopwords = set(STOP_WORDS)

# Cargar spaCy solo una vez y sin funciones lentas
nlp = spacy.load("en_core_web_lg")

def top_words (df, n=20):
    """
    Esta función devuelve las n palabras más comunes en un dataframe de pandas que contiene reviews de texto en una columna llamada 'reviewText'
    
    df: Es el dataset o corpus que informamos a la función. Debe contener la columna 'reviewText'
    n: Es el número de palabras más comunes que queremos obtener. Por defecto es 20, pero podemos ajustarlo a nuestro gusto.
    return: Una lista de tuplas con las n palabras más comunes y su frecuencia.
    """
    all_words = []
    for review in df['reviewText']:
        words = review.split()
        all_words.extend(words)
    # contador global
    wf = Counter(all_words)
    wf_most_common = wf.most_common(n)
    return wf_most_common

def remove_stopwords(text, stopwords):
    """
    Esta función elimina las stopwords de un texto dado.
    text: Es el texto del cual queremos eliminar las stopwords.
    return: El texto sin stopwords.
    """
    tokens = text.split()   # ya está limpio
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)

def preprocess_text (dataset):
    """
    Esta función realiza un preprocesamiento básico del texto en la columna 'reviewText' de un dataframe de pandas.
    Convierte el texto a minúsculas y elimina los caracteres que no son letras ni números. 
    
    dataset: Es el dataset o corpus que informamos a la función. Debe contener la columna 'reviewText'
    return: El dataframe con la columna 'reviewText' preprocesada.
    """
    dataset['reviewText'] = dataset['reviewText'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.lower()))
    return dataset


def top_ngrams(df, n=20, ngram=2):
    """
    Esta función devuelve los n-grams más comunes en un dataframe de pandas que contiene reviews de texto en una columna llamada 'reviewText'
    
    df: Es el dataset o corpus que informamos a la función. Debe contener la columna 'reviewText'
    n: Es el número de n-grams más comunes que queremos obtener. Por defecto es 20, pero podemos ajustarlo a nuestro gusto.
    ngram: Es el tamaño del n-grama que queremos obtener. Por defecto es 2 (bigramas), pero podemos ajustarlo a nuestro gusto.
    return: Una lista de tuplas con los n-grams más comunes y su frecuencia.
    """
    all_ngrams = []
    for review in df['reviewText']:
        words = review.split()
        ngrams = zip(*[words[i:] for i in range(ngram)])
        ngrams = [' '.join(ngram) for ngram in ngrams]
        all_ngrams.extend(ngrams)

def preprocess_text_to_tokens(text):
    """
    Esta función realiza un preprocesamiento básico del texto dado.
    Convierte el texto a minúsculas, elimina los caracteres que no son letras ni números, y divide el texto en tokens (palabras).
    text: Es el texto que informamos a la función.
    return: Una lista de tokens (palabras) preprocesadas.
    """
    import re
    text['reviewText'] = text['reviewText'].apply(
        lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x).lower()) if pd.notnull(x) else ""
    )
    return text


def tokenize(text):
    """ 
    Esta función divide el texto en tokens (palabras).
    text: Es el texto que informamos a la función."""
    return text.split()

def lemmatize_text(text):
    """
    Esta función lematiza el texto dado utilizando spaCy.
    text: Es el texto que informamos a la función.
    return: El texto lematizado.
    """
    doc = nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc])
    return lemmatized

def remove_entities(text):
    """
    Recibe un string, aplica NER con spaCy y elimina las entidades
    pertenecientes a REMOVE_LABELS. Devuelve un string limpio.
    """
    import spacy

    # Cargar spaCy
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    #definir etiquetas a eliminar
    REMOVE_LABELS = {
        "ORG", "PERSON", "GPE", "PRODUCT",
        "LOC", "WORK_OF_ART", "FAC", "LAW"
    }

    # Convertir a string (por seguridad)
    text = str(text)

    # Procesamos el texto
    doc = nlp(text)

    # Identificar spans a eliminar
    spans_to_remove = []
    for ent in doc.ents:
        if ent.label_ in REMOVE_LABELS:
            spans_to_remove.append((ent.start_char, ent.end_char))

    # Si no hay entidades → devolvemos el texto normalizado
    if not spans_to_remove:
        return " ".join(text.split())

    # Construir el texto sin las entidades
    cleaned_parts = []
    last_end = 0

    for start, end in spans_to_remove:
        cleaned_parts.append(text[last_end:start])
        last_end = end

    cleaned_parts.append(text[last_end:])
    cleaned_text = "".join(cleaned_parts)

    # Normalizar espacios antes de devolver
    return " ".join(cleaned_text.split())





