import os
from collections import defaultdict
from search_backend.tokenizer_stemmer import TokenizerStemmer, process_dir, Tokenizer
from nltk import ngrams
from nltk.tokenize import sent_tokenize


class NGramModel:
    def __init__(self, n, language):
        self.frequences = defaultdict(int)
        self.frequences_primary = defaultdict(int)

        self.probabilities = defaultdict(float)
        self.tokenizer_stemmer = Tokenizer()
        self.n = n
        self.language = language

    def process_dir(self, root_path, encoding="utf-8"):
        for text in process_dir(root_path, encoding):
            self.process_text(text)

        for n_gram in self.frequences:
            self.probabilities[n_gram] = float(self.frequences[n_gram]) / self.frequences_primary[n_gram[:-1]]
        return self

    def process_text(self, text):
        for _, _, stemmed_word in self.tokenizer_stemmer.span_tokenize(text):
            n_grams = self.get_n_grams(stemmed_word)
            for n_gram in n_grams:
                self.frequences[n_gram] += 1
                self.frequences_primary[n_gram[:-1]] += 1

    def get_n_grams(self, word):
        word = '^' + word + '$'
        return ngrams(word, self.n)

    def get_probability_query(self, query):
        probability = 1.0
        for _, _, stemmed_word in self.tokenizer_stemmer.span_tokenize(query):
            n_grams = self.get_n_grams(stemmed_word)
            for n_gram in n_grams:
                if n_gram in self.probabilities:
                    probability *= self.probabilities[n_gram]
                else:
                    probability = 0
        return probability

    @staticmethod
    def get_language_of_a_query(query, list_of_models):
        prob = []
        for model in list_of_models:
            prob.append((model.get_probability_query(query), model.language))

        return max(prob, key=lambda x: x[0])[1]


class NGramWordsModel:
    def __init__(self, n, language):
        self.frequences = defaultdict(int)
        self.frequences_primary = defaultdict(int)

        self.probabilities = defaultdict(float)
        self.probabilities_primary = defaultdict(float)
        self.tokenizer = Tokenizer()
        self.n = n
        self.language = language
        self.gamma = 0.9
        self.betta = 1 - 10 ** -2

    def process_dir(self, root_path, encoding="utf-8"):
        for text in process_dir(root_path, encoding):
            self.process_text(text)

        for n_gram in self.frequences:
            self.probabilities[n_gram] = float(self.frequences[n_gram]) / self.frequences_primary[n_gram[0]]

        sum_all_words = sum(self.frequences_primary.values())
        for word in self.frequences_primary:
            self.probabilities_primary[word] = float(self.frequences_primary[word]) / sum_all_words
        return self

    def process_text(self, text):
        for sentence in sent_tokenize(text):
            token_words = ['<<sent_begin>>'] + list(self.tokenizer.tokenize(sentence)) + ['<<sent_end>>']

            for first_word, second_word in zip(token_words, token_words[1:]):
                self.frequences[(first_word, second_word)] += 1
                self.frequences_primary[first_word] += 1

    def get_probability_query(self, query):
        probability = 1.0
        for sentence in sent_tokenize(query):
            token_words = ['<<sent_begin>>'] + list(self.tokenizer.tokenize(sentence)) + ['<<sent_end>>']

            for first_word, second_word in zip(token_words, token_words[1:]):

                probability *= self.betta * (self.probabilities[(first_word, second_word)] * self.gamma \
                                             + self.probabilities_primary[first_word] * (1 - self.gamma)) \
                               + (1 - self.betta)

        return probability

    @staticmethod
    def get_language_of_a_query(query, list_of_models):
        prob = []
        for model in list_of_models:
            prob.append((model.get_probability_query(query), model.language))
        print(prob)
        return max(prob, key=lambda x: x[0])[1]
