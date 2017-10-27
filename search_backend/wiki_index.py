import os
from collections import defaultdict
import pickle
import math
import bz2
import xml.etree.ElementTree as ET
from lxml import etree
import re
from search_backend.tokenizer_stemmer import TokenizerStemmer


class WikiPageIndex:
    def __init__(self, file_id, xml):
        self.file_id = file_id
        self.xml = xml
        self.body_index = BodyIndex()
        self.anchor_index = TextIndex()
        self.title_index = TextIndex()
        self.links = None

    def build_index(self):
        self.links = self.xml.xpath("//text()")
        self.body_index.process_text(self.get_text(), self.file_id)
        self.title_index.process_text(self.get_text(), self.file_id)
        self.anchor_index.process_text(self.get_links_str(), self.file_id)
        return self

    def get_links_str(self):
        return re.findall(r"\[\[(.*?)\]\]", ' '.join(self.links))

    def get_text(self):
        return self.xml.xpath("//text")[0].text

    def get_title(self):
        return self.xml.xpath("//title")[0].text

    def get_features(self):
        raise NotImplementedError


class TextIndex:
    tokenizer_stemmer = TokenizerStemmer()
    word_occurrences = defaultdict(dict)
    word_count = defaultdict(dict)
    cache_next = defaultdict(dict)
    cache_prev = defaultdict(dict)

    def __init__(self):
        self.features = [0] * 136

    @classmethod
    def process_text(cls, text, file_id):
        """Build index for the file"""
        for start, end, stemmed_word in cls.tokenizer_stemmer.span_tokenize(text):
            if file_id not in cls.word_occurrences[stemmed_word]:
                cls.word_occurrences[stemmed_word][file_id] = [start]
                cls.word_count[file_id][stemmed_word] = 1
            else:
                cls.word_occurrences[stemmed_word][file_id].append(start)
                cls.word_count[file_id][stemmed_word] += 1

    @classmethod
    def binary_search(cls, term, file_id, low, high, current, mode=True):
        occurrence_list = cls.word_occurrences[term][file_id]

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

    @classmethod
    def next(cls, term, file_id, current):
        """Get a next occurrence of a term in the document after the current position"""
        occurrence_list = cls.word_occurrences[term][file_id]
        curr_cache = cls.cache_next[term][file_id] if file_id in cls.cache_next[term] else 0

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
        curr_cache = cls.binary_search(term, file_id, low, high, current)
        return occurrence_list[curr_cache]

    @classmethod
    def prev(cls, term, file_id, current):
        """Get a previous occurrence of a term in the document before the current position"""
        occurrence_list = cls.word_occurrences[term][file_id]
        curr_cache = cls.cache_prev[term][file_id] if file_id in cls.cache_prev[term] else 0
        if len(occurrence_list) == 0 or occurrence_list[0] >= current:
            return None

        if occurrence_list[-1] < current:
            curr_cache = len(occurrence_list) - 1
            return occurrence_list[curr_cache]
        if curr_cache < len(occurrence_list) - 1 and occurrence_list[curr_cache + 1] >= current:
            high = curr_cache + 1
        else:
            high = len(occurrence_list) - 1
        jump = 1
        low = high - jump
        while low > 0 and occurrence_list[low] >= current:
            high = low
            jump *= 2
            low = high - jump
        if low <= 0:
            low = 0
        curr_cache = cls.binary_search(term, file_id, low, high, current, False)
        return occurrence_list[curr_cache]

    @classmethod
    def next_phrase(cls, terms, file_id, position, precision):
        """Find the position of a phrase in the document"""
        cur_pos = position
        for term in terms:
            cur_pos = cls.next(term, file_id, cur_pos)
            if not cur_pos:
                return None
        back_pos = cur_pos

        for term in terms[:-1:][::-1]:
            back_pos = cls.prev(term, file_id, back_pos)
        if cur_pos - back_pos <= precision + precision / 2:
            return back_pos, cur_pos
        else:
            return cls.next_phrase(terms, file_id, back_pos + 1, precision)

    @classmethod
    def tf_idf_scoring(cls, query_terms, documents_id):
        idf = defaultdict(float)
        tf_idf_vectors = defaultdict(list)
        query_tf_idf_vector = []
        query_vector_len = 0
        query_tf = defaultdict(int)

        for term in query_terms:
            query_tf[term] += 1

        words_index = set()
        for document_id in documents_id:
            words_index = words_index.union(set(cls.word_count[document_id].keys()))

        for word in words_index:
            idf[word] = len(set(cls.word_occurrences[word].keys()).intersection(set(documents_id)))
            query_tf_idf_vector.append(query_tf[word] * math.log(len(documents_id) / (1 + idf[word])))
            query_vector_len += query_tf_idf_vector[-1] ** 2

        for document_id in documents_id:
            for word in words_index:
                tf = cls.word_count[document_id][word] if word in cls.word_count[document_id] else 0
                tf_idf_vectors[document_id] \
                    .append(tf * math.log(len(documents_id) / (1 + idf[word])))

        documents_rank = []

        for document_id in documents_id:
            dot_product = 0
            document_vector_len = 0

            for x_index, query_tf_idf in enumerate(query_tf_idf_vector):
                dot_product += query_tf_idf * tf_idf_vectors[document_id][x_index]
                document_vector_len += tf_idf_vectors[document_id][x_index] ** 2

            cosine = dot_product / (document_vector_len * query_vector_len)

            documents_rank.append((cosine, document_id))
        return sorted(documents_rank, key=lambda x: x[0])

    @classmethod
    def get_docs_query_intersection(cls, terms):
        """Get all documents id which consist all the query terms"""
        if len(terms) == 0:
            return set()

        docs_ids = set(cls.word_occurrences[terms[0]].keys())
        for term in terms[1:]:
            docs_ids.intersection_update(cls.word_occurrences[term].keys())
        return docs_ids


class BodyIndex(TextIndex):
    word_occurrences = defaultdict(dict)
    word_count = defaultdict(dict)
    cache_next = defaultdict(dict)
    cache_prev = defaultdict(dict)


class TitleIndex(TextIndex):
    word_occurrences = defaultdict(dict)
    word_count = defaultdict(dict)
    cache_next = defaultdict(dict)
    cache_prev = defaultdict(dict)


class AnchorIndex(TextIndex):
    word_occurrences = defaultdict(dict)
    word_count = defaultdict(dict)
    cache_next = defaultdict(dict)
    cache_prev = defaultdict(dict)


class WikiIndex:
    def __init__(self):
        self.pages = dict()
        self.last_file_id = 0
        self.scoring_model = ScoringModel()

    @staticmethod
    def load_index(path='text_index.pkl'):
        with open(path, 'rb') as g:
            return pickle.load(g)

    def save_index(self, path='text_index.pkl'):
        with open(path, 'wb') as g:
            pickle.dump(self, g)

    def build_index_root(self, root_path):
        """Build index for all files in the directory"""
        for dirName, subdirList, fileList in os.walk(root_path):
            for file_name in fileList:
                if file_name.endswith('.bz2'):
                    self.build_index_file(os.path.join(dirName, file_name))
        return self

    def build_index_file(self, file_path):
        """Build index for one file"""
        for xml in self.read_wiki_file(file_path):
            self.pages[self.last_file_id] = WikiPageIndex(xml=xml, file_id=self.last_file_id)
            self.last_file_id += 1
        self.save_index()
        return self

    @staticmethod
    def read_wiki_file(file_path):
        with bz2.open(file_path) as f:
            page = ''
            for line in f:
                line = str(line, encoding='utf-8').strip()
                if line == '<page>':
                    page = line
                elif line == '</page>':
                    page += '\n' + line
                    yield etree.fromstring(page)
                elif page != '':
                    page += '\n' + line

    def search_query(self, query):
        """Find relevant documents for the query"""
        matched_ids = []
        matched_in_text = dict()

        query_terms = TextIndex.tokenizer_stemmer.tokenize(query)

        found_docs = BodyIndex.get_docs_query_intersection(query_terms)

        docs_features = WikiPageIndex.get_features(found_docs)

        relevant_docs = self.scoring_model.predict(docs_features)

        for document_id in relevant_docs:
            found = BodyIndex.next_phrase(query_terms, document_id, 0, len(query))
            if found:
                matched_ids.append(document_id)
                matched_in_text[document_id] = found

    def build_query_features(self):
        pass


class ScoringModel:
    def predict(self, features):
        raise NotImplementedError


if __name__ == '__main__':
    wiki_files = WikiIndex().build_index_file('../../enwiki-20171020-pages-articles1.xml-p10p30302.bz2')
