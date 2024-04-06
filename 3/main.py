import os
import re
from collections import defaultdict
from multiprocessing import Pool

from nltk import word_tokenize
from nltk.corpus import stopwords
from pymystem3 import Mystem

RUSSIAN_STOPWORDS = set(stopwords.words('russian'))


def is_valid_word(lemma: str) -> bool:
    return re.match(r'^[а-яА-ЯёЁa-zA-Z]+-?[а-яА-ЯёЁa-zA-Z]*$', lemma) is not None


def process_file(directory: str, file_name: str) -> (str, list[str]):
    file_path = os.path.join(directory, file_name)
    file_lemmas = []
    mystem = Mystem()

    print(f'Processing file: {file_path}')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        tokens = word_tokenize(text, language="russian")
        for word in tokens:
            lemma: str = mystem.lemmatize(word)[0]
            if lemma not in RUSSIAN_STOPWORDS and is_valid_word(lemma):
                file_lemmas.append(lemma)

    return file_name, file_lemmas


def make_inverted_index(source_dir: str) -> defaultdict[str, set[str]]:
    inverted_index: defaultdict[str, set[str]] = defaultdict(set)
    source_files = os.listdir(source_dir)

    with Pool() as pool:
        file_lemmas_pairs = pool.starmap(
            process_file,
            [(source_dir, file_name) for file_name in source_files]
        )

    for file_name, lemmas in file_lemmas_pairs:
        for lemma in lemmas:
            inverted_index[lemma].add(file_name)

    return inverted_index


def boolean_search(query: str, inverted_index: defaultdict[str, set[str]]) -> set[str]:
    mystem = Mystem()
    eval_expression = ""

    # Множество, состоящее из названий всех файлов
    _all = set(os.listdir('/Users/ruslan/Desktop/ИТИС/informational-search/1/files'))

    for word in query.split():
        if word in ["&", "|"]:
            eval_expression += f" {word} "
        else:
            # Если у слова есть приставка !, то берём весь словарь целиком (_all) и вычитаем оттуда наш
            if word[0] == "!":
                lemma: str = mystem.lemmatize(word[1:])[0]
                if lemma in inverted_index:
                    eval_expression += f"(_all - set(inverted_index['{lemma}']))"
                else:
                    eval_expression += "_all"
            # Для слов без приставки добавляем множество файлов для леммы, если она есть в индексе, иначе - пустое множество
            else:
                lemma: str = mystem.lemmatize(word)[0]
                if lemma in inverted_index:
                    eval_expression += f"set(inverted_index['{lemma}'])"
                else:
                    eval_expression += "set()"

    try:
        return eval(eval_expression)
    except SyntaxError:
        return set()


def save_to_file(inverted_index: defaultdict[str, set[str]], queries: list[str]):
    with open('/Users/ruslan/Desktop/ИТИС/informational-search/3/boolean_search.txt', 'w', encoding='utf-8') as file:
        file.write("Boolean search:\n")
        for query in queries:
            files_names = sorted(boolean_search(query, inverted_index))
            file.write(f"Query - {query}\n")
            file.write(f"Result - {', '.join(files_names) if files_names else 'No files'}\n\n")

    with open('/Users/ruslan/Desktop/ИТИС/informational-search/3/inverted_index.txt', 'w', encoding='utf-8') as file:
        file.write("Inverted index:\n")
        for word, files_names in inverted_index.items():
            sorted_files_names = sorted(
                [file_name.split('.')[0] for file_name in files_names]
            )
            file.write(f"{word} - {', '.join(sorted_files_names)}\n")


if __name__ == "__main__":
    inverted_index = make_inverted_index('/Users/ruslan/Desktop/ИТИС/informational-search/1/files')
    word1, word2, word3 = "манипулировать", "середина", "декодирование"
    queries = [
        f"{word1} &  {word2} |  {word3}",
        f"{word1} & !{word2} | !{word3}",
        f"{word1} |  {word2} |  {word3}",
        f"{word1} | !{word2} | !{word3}",
        f"{word1} &  {word2} &  {word3}"
    ]
    save_to_file(inverted_index, queries)
