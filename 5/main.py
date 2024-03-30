import math
import os
from collections import Counter

import pandas as pd
from four.main import compute_tf, compute_idf, compute_tf_idf
from pymystem3 import Mystem
from sklearn.metrics.pairwise import cosine_similarity

mystem = Mystem()


def query_to_vector(query: str, idf: dict[str, float], total_docs_count: int):
    """
    Преобразует запрос в вектор TF-IDF, с применением лемматизации к каждому слову.
    """
    words = mystem.lemmatize(query.lower())
    words = [word for word in words if word.strip() and word not in [' ', '\n']]
    query_length = len(words)
    words_counter = Counter(words)
    query_vector: dict[str, float] = {}

    for word, count in words_counter.items():
        tf = count / query_length
        # Если слово не в IDF, используем минимальный IDF
        word_idf = idf.get(word, math.log10(total_docs_count))
        query_vector[word] = tf * word_idf

    return query_vector


def compute_cosine_similarity(doc_vector: dict[str, float], query_vector:  dict[str, float]) -> float:
    """
    Вычисляет косинусное сходство между двумя векторами.
    """
    all_words = list(set(doc_vector.keys()).union(set(query_vector.keys())))
    doc_vector = [doc_vector.get(word, 0) for word in all_words]
    query_vector = [query_vector.get(word, 0) for word in all_words]
    return cosine_similarity([doc_vector], [query_vector])[0][0]


def vector_search_multi(queries: [str], tf_idf: dict[str, dict[str, float]], idf: dict[str, float],
                        total_docs_count: int, output_txt_file: str, output_csv_file: str):
    results = []
    with open(output_txt_file, 'w') as txt_file:
        for query in queries:
            query_vector = query_to_vector(query, idf, total_docs_count)
            scores: dict[str, float] = {}
            for doc, doc_vector in tf_idf.items():
                scores[doc] = compute_cosine_similarity(doc_vector, query_vector)

            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            txt_file.write(f"Query: {query}\n")
            for doc, score in sorted_scores:
                if score == 0:
                    break
                result_line = f"Document: {doc}, Score: {score}\n"
                txt_file.write(result_line)
                results.append({"Query": query, "Document": doc, "Score": score})
            txt_file.write("\n")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_file, index=False)
    print(f"Результаты поиска сохранены в {output_txt_file} и {output_csv_file}.")


if __name__ == '__main__':
    path = '/Users/ruslan/Desktop/ИТИС/informational-search/2/files'
    corpus = {}
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as f:
            corpus[file] = f.read().split()

    tf = {doc: compute_tf(text) for doc, text in corpus.items()}
    idf = compute_idf(corpus)
    tf_idf = compute_tf_idf(tf, idf)

    queries = ['манипулировать', 'манипулировать середина', 'манипулировать середина декодирование']
    output_txt = 'vector_search.txt'
    output_csv = 'vector_search.csv'
    vector_search_multi(queries, tf_idf, idf, len(corpus), output_txt, output_csv)
