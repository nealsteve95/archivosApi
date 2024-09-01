import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Cargar el archivo CSV
df = pd.read_csv('EdX.csv')

# Preprocesar el texto
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

df['Processed_Description'] = df['Description'].apply(preprocess)

# Generar la matriz TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Processed_Description'])

# Calcular la similitud del coseno
similarity_matrix = cosine_similarity(tfidf_matrix)

# Identificar palabras clave
feature_names = vectorizer.get_feature_names_out()

# Función para obtener palabras clave y su similitud
def get_keywords_with_similarity(book_index, top_n=10):
    # Obtener la matriz de similitud para el libro en particular
    similarity_scores = similarity_matrix[book_index]
    keywords_scores = {}

    # Iterar sobre las características y calcular la similitud
    for word_index in range(len(feature_names)):
        word = feature_names[word_index]
        word_score = tfidf_matrix[:, word_index].toarray()
        similarity_sum = np.sum(word_score * similarity_scores[:, np.newaxis])
        keywords_scores[word] = similarity_sum.mean()

    # Ordenar palabras clave por similitud
    sorted_keywords = sorted(keywords_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]

# Obtener las 10 palabras clave más similares al libro en el índice 0
keywords_with_similarity = get_keywords_with_similarity(0)

# Convertir el resultado a JSON
result = {word: float(similarity) for word, similarity in keywords_with_similarity}
print(json.dumps(result))