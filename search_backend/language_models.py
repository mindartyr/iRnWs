import os
from collections import defaultdict
from search_backend.tokenizer_stemmer import TokenizerStemmer, process_dir, Tokenizer
from nltk import ngrams
from nltk.tokenize import sent_tokenize

LETTERS= 'abcdefghijklmnopqrstuvwxyz'


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

    def get_probability_word(self, word):
        probability = 1.0
        n_grams = self.get_n_grams(word)
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

    @staticmethod
    def _word_one_edit(word):
        """All edits that are one edit away from `word`."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        # deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        # replaces = [L + c + R[1:] for L, R in splits if R for c in LETTERS]
        # inserts = [L + c + R for L, R in splits for c in LETTERS]
        return set(transposes)  # + replaces + inserts + deletes)

    @staticmethod
    def _word_two_edits(word):
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in NGramModel._word_one_edit(word) for e2 in NGramModel._word_one_edit(e1))

    def spell_check(self, query):

        suggested_query = []

        for word in self.tokenizer_stemmer.tokenize(query):
            prob = self.get_probability_word(word)
            max_probability = prob
            max_probability_original = max_probability
            corresponding_word = word

            for edited_word in self._word_one_edit(word):
                prob = self.get_probability_word(edited_word)
                if prob > max_probability:
                    max_probability = prob
                    corresponding_word = edited_word

            # for edited_word in self._word_two_edits(word):
            #     prob = self.get_probability_word(edited_word)
            #     if prob > max_probability:
            #         max_probability = prob
            #         corresponding_word = edited_word

            suggested_query.append(corresponding_word if max_probability > 1.5 * max_probability_original else word)
        return ' '.join(suggested_query)


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
        return max(prob, key=lambda x: x[0])[1]
