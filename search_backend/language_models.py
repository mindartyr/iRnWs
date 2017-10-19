import os
from collections import defaultdict
from search_backend.tokenizer_stemmer import TokenizerStemmer
from nltk import ngrams


class NGramModel:
    def __init__(self, n):
        self.frequences = defaultdict(int)
        self.frequences_primary = defaultdict(int)

        self.probabilities = defaultdict(float)
        self.tokenizer_stemmer = TokenizerStemmer()
        self.n = n

    def process_dir(self, root_path):
        for dirName, subdirList, fileList in os.walk(root_path):
            for file_name in fileList:
                if file_name.endswith('.txt'):
                    with open(os.path.join(dirName, file_name)) as f:
                        text = f.read()
                        self.process_text(text)
        return self

    def process_text(self, text):
        for _, _, stemmed_word in self.tokenizer_stemmer.tokenize_and_stem(text):
            n_grams = self.get_n_grams(stemmed_word)
            for n_gram in n_grams:
                self.frequences[n_gram] += 1
                self.frequences_primary[n_gram[:-1]] += 1

        for n_gram in self.frequences:
            self.probabilities[n_gram] = float(self.frequences[n_gram]) / self.frequences_primary[n_gram[:-1]]

    def get_n_grams(self, word):
        word = '^' + word + '$'
        return ngrams(word, self.n)

    def get_probability_query(self, query):
        probability = 1.0
        for _, _, stemmed_word in self.tokenizer_stemmer.tokenize_and_stem(query):
            n_grams = self.get_n_grams(stemmed_word)
            for n_gram in n_grams:
                if n_gram in self.probabilities:
                    probability *= self.probabilities[n_gram]
        return probability



