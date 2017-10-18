import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/python-3.6.3-docs-text')


class TextIndex:

    def __init__(self):
        self.word_occurrences = defaultdict(dict)
        self.files_ids = defaultdict(str)
        self.cache_next = defaultdict(dict)
        self.eng_stop_words = set(stopwords.words('english'))
        self.eng_stemmer = SnowballStemmer("english")
        self.tokenizer = RegexpTokenizer(r'\w+')

    def build_index(self, root_path):
        curr_file_id = 0
        for dirName, subdirList, fileList in os.walk(root_path):
            for file_name in fileList:
                if file_name.endswith('.txt'):
                    self.files_ids[curr_file_id] = file_name
                    curr_file_id += 1

                    with open(os.path.join(ROOT_PATH, dirName, file_name)) as f:
                        self.process_text(f.read(), curr_file_id)
        return self

    def process_text(self, text, file_id):
        for start, end, stemmed_word in self.tokenize_and_stem(text):
            if file_id not in self.word_occurrences[stemmed_word]:
                self.word_occurrences[stemmed_word][file_id] = [(start, end)]
            else:
                self.word_occurrences[stemmed_word][file_id].append((start, end))

    def process_query(self, query):
        for start, end, stemmed_word in self.tokenize_and_stem(query):
            pass

    def tokenize_and_stem(self, text):
        for start, end in self.tokenizer.span_tokenize(text):
            word = text[start:end].lower()
            stemmed_word = self.eng_stemmer.stem(word)
            if word not in self.eng_stop_words:
                yield start, end, stemmed_word

    def binary_search(self, term, file_id, low, high, current):
        occurrence_list = self.word_occurrences[term][file_id]

        while high - low > 1:
            middle = int((low + high) / 2)
            if occurrence_list[middle] <= current:
                low = middle
            else:
                high = middle
        return high

    def next(self, term, file_id, current):
        occurrence_list = self.word_occurrences[term][file_id]
        curr_cache = self.cache_next[term][file_id]
        if len(occurrence_list) == 0 or occurrence_list[-1] <= current:
            return None
        if occurrence_list[0] > current:
            curr_cache = 0
            return occurrence_list[curr_cache]
        if curr_cache > 0 and occurrence_list[curr_cache - 1] <= current:
            low = curr_cache - 1
        else:
            low = 1
        jump = 1
        high = low + jump
        while high < len(occurrence_list) - 1 and occurrence_list[high] <= current:
            low = high
            jump *= 2
            high = low + jump
        if high > len(occurrence_list) - 1:
            high = len(occurrence_list) - 1
        curr_cache = self.binary_search(term, file_id, low, high, current)
        return occurrence_list[curr_cache]

    def next_phrase(self, terms, file_id, position):
        cur_pos = position
        for term in terms:
            cur_pos = self.next(term, file_id, cur_pos)
        if not cur_pos:
            return None
        back_pos =

