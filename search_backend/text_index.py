import os
from collections import defaultdict
import pickle
from search_backend.tokenizer_stemmer import TokenizerStemmer

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/python-3.6.3-docs-text')


class TextIndex:

    def __init__(self):
        self.word_occurrences = defaultdict(dict)
        self.original_text = dict()
        self.files_ids = defaultdict(str)
        self.cache_next = defaultdict(dict)
        self.cache_prev = defaultdict(dict)
        self.tokenizer_stemmer = TokenizerStemmer()

    @staticmethod
    def load_index(path='text_index.pkl'):
        with open(path, 'rb') as g:
            return pickle.load(g)

    def save_index(self, path='text_index.pkl'):
        with open(path, 'wb') as g:
            pickle.dump(self, g)

    def build_index(self, root_path):
        curr_file_id = 0
        for dirName, subdirList, fileList in os.walk(root_path):
            for file_name in fileList:
                if file_name.endswith('.txt'):
                    self.files_ids[curr_file_id] = file_name
                    curr_file_id += 1

                    with open(os.path.join(dirName, file_name)) as f:
                        text = f.read()
                        self.original_text[curr_file_id] = text
                        self.process_text(text, curr_file_id)
        self.save_index()
        return self

    def process_text(self, text, file_id):
        for start, end, stemmed_word in self.tokenizer_stemmer.tokenize(text):
            if file_id not in self.word_occurrences[stemmed_word]:
                self.word_occurrences[stemmed_word][file_id] = [start]
            else:
                self.word_occurrences[stemmed_word][file_id].append(start)

    def process_query(self, query):
        terms = []
        matched_docs = []

        for _, _, stemmed_word in self.tokenizer_stemmer.tokenize(query):
            terms.append(stemmed_word)

        for document_id in self.get_docs_for_and_query(terms):
            found = self.next_phrase(terms, document_id, 0, len(query))
            if found:
                matched_docs.append({
                    'title': self.files_ids[document_id],
                    'snippet': self.original_text[document_id][found[0]:found[1]+20],
                    'href': 'http://www.{0}.com'.format(document_id)
                })

        return matched_docs

    def process_conditional_query(self, query):
        or_queries = query.split(' OR ')
        results = []
        for or_query in or_queries:
            results += 1
        raise NotImplementedError()

    def get_docs_for_and_query(self, terms):
        if len(terms) == 0:
            return set()

        docs_ids = set(self.word_occurrences[terms[0]].keys())
        for term in terms[1:]:
            docs_ids.intersection_update(self.word_occurrences[term].keys())

        return docs_ids

    def binary_search(self, term, file_id, low, high, current, mode=True):
        occurrence_list = self.word_occurrences[term][file_id]

        while high - low > 1:
            middle = int((low + high) / 2)
            if mode:
                if occurrence_list[middle] <= current:
                    low = middle
                else:
                    high = middle
            else:
                if occurrence_list[middle] < current:
                    low = middle
                else:
                    high = middle
        return high if mode else low

    def next(self, term, file_id, current):
        occurrence_list = self.word_occurrences[term][file_id]
        curr_cache = self.cache_next[term][file_id] if file_id in self.cache_next[term] else 0

        if len(occurrence_list) == 0 or occurrence_list[-1] <= current:
            return None
        if occurrence_list[0] > current:
            curr_cache = 0
            return occurrence_list[curr_cache]
        if curr_cache > 0 and occurrence_list[curr_cache - 1] <= current:
            low = curr_cache - 1
        else:
            low = 0
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

    def prev(self, term, file_id, current):
        occurrence_list = self.word_occurrences[term][file_id]
        curr_cache = self.cache_prev[term][file_id] if file_id in self.cache_prev[term] else 0
        if len(occurrence_list) == 0 or occurrence_list[0] >= current:
            return None

        if occurrence_list[-1] < current:
            curr_cache = len(occurrence_list)-1
            return occurrence_list[curr_cache]
        if curr_cache < len(occurrence_list)-1 and occurrence_list[curr_cache + 1] >= current:
            high = curr_cache + 1
        else:
            high = len(occurrence_list)-1
        jump = 1
        low = high - jump
        while low > 0 and occurrence_list[low] >= current:
            high = low
            jump *= 2
            low = high - jump
        if low <= 0:
            low = 0
        curr_cache = self.binary_search(term, file_id, low, high, current, False)
        return occurrence_list[curr_cache]

    def next_phrase(self, terms, file_id, position, precision):
        cur_pos = position
        for term in terms:
            cur_pos = self.next(term, file_id, cur_pos)
            if not cur_pos:
                return None
        back_pos = cur_pos

        for term in terms[:-1:][::-1]:
            back_pos = self.prev(term, file_id, back_pos)
        if cur_pos - back_pos <= precision + precision / 2:
            return back_pos, cur_pos
        else:
            return self.next_phrase(terms, file_id, back_pos+1, precision)


if __name__ == '__main__':
    if 0:
        f = open('text_index.pkl', 'wb')
        text_index = TextIndex().build_index('../data/')
        pickle.dump(text_index, f)
    else:
        f = open('text_index.pkl', 'rb')
        text_index = pickle.load(f)
    text_index.process_query('python language')
    f.close()
