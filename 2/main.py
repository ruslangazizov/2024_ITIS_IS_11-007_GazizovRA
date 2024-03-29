import os
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem

# nltk.download('stopwords')  # нужно запустить только один раз, чтобы скачать стоп-слова
RUSSIAN_STOPWORDS = set(stopwords.words('russian'))


def is_valid_word(token: str) -> bool:
    return re.match(r'^[а-яА-ЯёЁa-zA-Z]+-?[а-яА-ЯёЁa-zA-Z]*$', token) is not None


def process_file(directory: str, file_name: str) -> (str, list[str]):
    file_path = os.path.join(directory, file_name)
    file_lemmas = []
    mystem = Mystem()

    print(f'Processing file: {file_path}')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        tokens = word_tokenize(text, language="russian")
        for word in tokens:
            lemma = mystem.lemmatize(word)[0]
            if lemma not in RUSSIAN_STOPWORDS and is_valid_word(lemma):
                file_lemmas.append(lemma)

    return file_name, file_lemmas


def main(source_dir: str, output_dir: str) -> defaultdict[str, set[str]]:
    inverted_index: defaultdict[str, set[str]] = defaultdict(set)
    source_files = os.listdir(source_dir)
    Path(output_dir).mkdir(exist_ok=True)

    with Pool() as pool:
        file_lemmas_pairs = pool.starmap(
            process_file,
            [(source_dir, file_name) for file_name in source_files]
        )

    for file_name, lemmas in file_lemmas_pairs:
        output_file_path = os.path.join(output_dir, file_name)
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for lemma in lemmas:
                output_file.write(f"{lemma}\n")
                inverted_index[lemma].add(file_name)
        print(f'Saved file {output_file_path}')

    print(inverted_index)


if __name__ == "__main__":
    main('/Users/ruslan/Desktop/ИТИС/informational-search/1/files',
         '/Users/ruslan/Desktop/ИТИС/informational-search/2/files')
