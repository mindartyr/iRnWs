import bz2
import math
import os
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from lxml import etree

from search_backend.dto.sql_dict import WordCount, WordOccurrences, SimpleDict
from search_backend.scoring_model.relevance_model import ScoringModel
from search_backend.text_processing.tokenizer_stemmer import ToktokTokeniserStemmer

DIR = os.path.dirname(os.path.realpath(__file__))


def get_features_mapping(amount):
    output = []
    i = 1
    while len(output) < amount:
        if i % 5 != 0 and i % 5 != 4:
            output.append(i)
        i += 1
    return output


class WikiPageIndex:
    def __init__(self, file_id, xml, line_number):
        self.file_id = file_id
        self.xml = xml
        self.start = line_number
        self.body_index = BodyIndex()
        self.anchor_index = AnchorIndex()
        self.title_index = TitleIndex()
        self.links = None

    def build_index(self):
        TextIndex.start.set(self.file_id, self.start)
        self.links = self.get_links(self.xml.xpath("string()"))
        self.body_index.process_text(self.get_text(), self.file_id)
        self.title_index.process_text(self.get_title(), self.file_id)
        self.anchor_index.process_text(' '.join(self.links), self.file_id)
        return self

    @staticmethod
    def get_links(x):
        return re.findall(r"\[\[(.*?)\]\]", x)

    def get_text(self):
        return self.xml.xpath("//text")[0].text

    def get_title(self):
        return self.xml.xpath("//title")[0].text

    @staticmethod
    def get_features(query_terms, docs):
        all_features = []

        for doc_id in docs:
            doc_features = list()
            covered_term_number = WikiPageIndex.get_covered_term_number(doc_id, query_terms)
            stream_length = WikiPageIndex.get_stream_length(doc_id)
            # idf = WikiPageIndex.get_idf(docs, query_terms)
            tf = WikiPageIndex.get_tf(doc_id, query_terms)
            tf_idf_features = WikiPageIndex.get_tf_idf_features(query_terms, doc_id)
            doc_features += covered_term_number   # 1 2 3
            doc_features += [feature / len(query_terms) for feature in covered_term_number]  # 6 7 8
            doc_features += stream_length  # 11 12 13
            # doc_features += idf  # 16 17 18
            doc_features += [sum(feature) for feature in tf]  # 21 22 23
            doc_features += [min(feature) for feature in tf]  # 26 27 28
            doc_features += [max(feature) for feature in tf]  # 31 32 33
            doc_features += [np.mean(feature) for feature in tf]  # 36 37 38
            doc_features += [np.var(feature) for feature in tf]  # 41 42 43
            doc_features += [sum(feature) / length for feature, length in zip(tf, stream_length)]  # 46 47 48
            doc_features += [min(feature) / length for feature, length in zip(tf, stream_length)]  # 51 52 53
            doc_features += [max(feature) / length for feature, length in zip(tf, stream_length)]  # 56 57 58
            doc_features += [np.mean(feature) / length for feature, length in zip(tf, stream_length)]  # 61 62 63
            doc_features += [np.var(feature) / length for feature, length in zip(tf, stream_length)]  # 66 67 68
            doc_features += tf_idf_features # 71 72 73 76 77 78 81 82 83 86 87 88 91 92 93
            all_features.append(doc_features)

        # print(['{0}:{1}'.format(number, feature)
        #        for feature, number in
        #        zip(all_features[0], get_features_mapping(len(all_features[0])))])
        return all_features

    @staticmethod
    def get_covered_term_number(doc_id, query_terms):
        return [BodyIndex.get_query_term_intersection(doc_id, query_terms),
                AnchorIndex.get_query_term_intersection(doc_id, query_terms),
                TitleIndex.get_query_term_intersection(doc_id, query_terms)]

    @staticmethod
    def get_stream_length(doc_id):
        return [BodyIndex.get_stream_length(doc_id),
                AnchorIndex.get_stream_length(doc_id),
                TitleIndex.get_stream_length(doc_id)]

    @staticmethod
    def get_idf(docs, terms):
        return [BodyIndex.get_idf(docs, terms),
                AnchorIndex.get_idf(docs, terms),
                TitleIndex.get_idf(docs, terms)]

    @staticmethod
    def get_tf(doc_id, terms):
        return [BodyIndex.get_tf(doc_id, terms),
                AnchorIndex.get_tf(doc_id, terms),
                TitleIndex.get_tf(doc_id, terms)]

    @staticmethod
    def get_tf_idf_features(terms, doc_id):
        output_features = []
        body_features = BodyIndex.get_tf_idf_features(terms, doc_id)
        anchor_features = AnchorIndex.get_tf_idf_features(terms, doc_id)
        title_features = TitleIndex.get_tf_idf_features(terms, doc_id)

        for f1, f2, f3 in zip(body_features, anchor_features, title_features):
            output_features += [f1, f2, f3]
        return output_features


class TextIndex:
    stream_length = SimpleDict('stream_length')
    start = SimpleDict('start', bulk_limit=100)
    tokenizer_stemmer = ToktokTokeniserStemmer()
    word_occurrences = WordOccurrences('text')
    word_count = WordCount('text')
    cache_next = defaultdict(dict)
    cache_prev = defaultdict(dict)
    # TODO: 1) implement tf-idf as a numpy array
    # TODO: 2) make mapping for words str<->int
    idf = SimpleDict('idf')
    tf = defaultdict(dict)

    @classmethod
    def drop(cls):
        cls.stream_length.drop()
        cls.word_count.drop()
        cls.word_occurrences.drop()
        TextIndex.start.drop()
        cls.idf.drop()

    @classmethod
    def process_text(cls, text, file_id):
        """Build index for the file"""
        cls.stream_length.set(file_id, len(text))
        temp_word_count = defaultdict(int)
        for w_position, stemmed_word in cls.tokenizer_stemmer.tokenize(text):
            cls.word_occurrences.set(file_id, stemmed_word, w_position)
            temp_word_count[stemmed_word] += 1
        cls.word_count.insert(file_id, temp_word_count)

    @classmethod
    def binary_search(cls, term, file_id, low, high, current, mode=True):
        occurrence_list = list(cls.word_occurrences.select_positions(file_id, term))

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
        occurrence_list = list(cls.word_occurrences.select_positions(file_id, term))
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
        occurrence_list = list(cls.word_occurrences.select_positions(file_id, term))
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
            words_index = words_index.union(set(cls.word_count.select_words(document_id)))

        for word in words_index:
            idf[word] = len(set(cls.word_occurrences.select_documents(word)).intersection(set(documents_id)))
            query_tf_idf_vector.append(query_tf[word] * math.log(len(documents_id) / (1 + idf[word])))
            query_vector_len += query_tf_idf_vector[-1] ** 2

        for document_id in documents_id:
            for word in words_index:
                tf = cls.word_count.select_count(document_id, word) if word in cls.word_count.select_words(document_id)\
                    else 0
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
    def calc_idf(cls):
        for word in cls.word_occurrences.get_df():
            cls.idf.set(word['_id'], word['df'])

    @classmethod
    def get_docs_query_intersection(cls, terms):
        """Get all documents id which consist all the query terms"""
        if len(terms) == 0:
            return set()

        docs_ids = set(cls.word_occurrences.select_documents(terms[0]))

        for term in terms[1:]:
            docs_ids.intersection_update(cls.word_occurrences.select_documents(term))
        return docs_ids

    @classmethod
    def get_query_term_intersection(cls, doc_id, terms):
        count = 0
        for term in terms:
            positions_len = len(list(cls.word_occurrences.select_positions(doc_id, term)))
            if positions_len > 0:
                count += 1
        return count

    @classmethod
    def get_stream_length(cls, doc_id):
        return cls.stream_length.select_value(doc_id)

    @classmethod
    def get_idf(cls, docs, terms):
        # maybe there should be just sum of idf of terms
        idf = 0
        for doc_id in docs:
            for word in terms:
                count = cls.word_count.select_count(document=doc_id, word=word)
                if count and count > 0:
                    idf += 1

        doc_amount = len(docs)

        idf = math.log(doc_amount / (1 + idf))
        return idf

    @classmethod
    def get_tf(cls, doc_id, terms):
        tf = []
        for term in terms:
            count = cls.word_count.select_count(document=doc_id, word=term)
            tf.append(count if count else 0)
        return tf

    @classmethod
    def get_tf_idf_features(cls, terms, doc_id):
        # TODO this tf-idf is without zeros
        tf_idf = []
        doc_amount = cls.word_count.doc_amount()

        for word in terms:
            df = cls.idf.select_value(word)
            df = df if df else 0
            tf = cls.word_count.select_count(doc_id, word)
            tf = tf if tf else 0
            tf_idf.append(tf * math.log(doc_amount / (1 + df)))
        return [sum(tf_idf),
                min(tf_idf),
                max(tf_idf),
                np.mean(tf_idf),
                np.var(tf_idf)]

    @classmethod
    def execute_bulk(cls):
        cls.word_occurrences.execute_bulk()
        cls.word_count.execute_bulk()
        cls.stream_length.execute_bulk()
        TextIndex.start.execute_bulk()
        print('started idf calculation')
        cls.calc_idf()
        cls.idf.execute_bulk()


class BodyIndex(TextIndex):
    stream_length = SimpleDict('stream_length_body')
    word_occurrences = WordOccurrences('body')
    word_count = WordCount('body')
    cache_next = defaultdict(dict)
    cache_prev = defaultdict(dict)
    idf = SimpleDict('idf_body')
    tf = defaultdict(dict)


class TitleIndex(TextIndex):
    stream_length = SimpleDict('stream_length_title')
    word_occurrences = WordOccurrences('title')
    word_count = WordCount('title')
    cache_next = defaultdict(dict)
    cache_prev = defaultdict(dict)
    idf = SimpleDict('idf_title')
    tf = defaultdict(dict)


class AnchorIndex(TextIndex):
    stream_length = SimpleDict('stream_length_anchor')
    word_occurrences = WordOccurrences('anchor')
    word_count = WordCount('anchor')
    cache_next = defaultdict(dict)
    cache_prev = defaultdict(dict)
    idf = SimpleDict('idf_anchor')
    tf = defaultdict(dict)


class WikiIndex:
    def __init__(self, path):
        self.dump_path = path
        self.last_file_id = 0
        self.scoring_model = ScoringModel().load_model()

    def build_index_root(self, root_path):
        """Build index for all files in the directory"""
        for dirName, subdirList, fileList in os.walk(root_path):
            for file_name in fileList:
                if file_name.endswith('.bz2'):
                    self.build_index_file(os.path.join(dirName, file_name))
        return self

    def build_index_file(self, file_name=None):
        """Build index for one file"""
        if not file_name:
            file_name = self.dump_path
        BodyIndex.drop()
        AnchorIndex.drop()
        TitleIndex.drop()
        for xml, line_number, page_id in tqdm(self.read_wiki_file(file_name)):
            WikiPageIndex(xml=xml, file_id=page_id, line_number=line_number).build_index()
            self.last_file_id += 1
            if self.last_file_id > 2000:
                break

        BodyIndex.execute_bulk()
        TitleIndex.execute_bulk()
        AnchorIndex.execute_bulk()
        return self

    @staticmethod
    def read_wiki_file_bz2(file_path):
        """Read wiki documents from bz2 wiki dump"""
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

    @staticmethod
    def read_wiki_file(file_path):
        """Read wiki documents from wiki dump"""
        beginning = 0
        page_id = 0
        with open(file_path) as f:
            page = ''
            for line_number, line in enumerate(f):
                line = line.strip()
                if line == '<page>':
                    beginning = line_number
                    page = line
                elif line == '</page>':
                    page += '\n' + line
                    yield etree.fromstring(page), beginning, page_id
                    page_id += 1
                elif page != '':
                    page += '\n' + line

    @staticmethod
    def get_xml(file_path, number):
        with open(file_path) as f:
            page = ''
            for line_number, line in enumerate(f):
                if line_number < number:
                    continue
                line = line.strip()
                if line == '<page>':
                    page = line
                elif line == '</page>':
                    page += '\n' + line
                    return etree.fromstring(page)
                elif page != '':
                    page += '\n' + line

    def search_query(self, query):
        """Find relevant documents for the query"""
        matched_ids = []
        matched_in_text = dict()

        query_terms = [word for _, word in TextIndex.tokenizer_stemmer.tokenize(query)]

        found_docs = BodyIndex.get_docs_query_intersection(query_terms)
        print('Length of first filtering: ', len(found_docs))

        for document_id in found_docs:
            found = BodyIndex.next_phrase(query_terms, document_id, 0, len(query))
            if found:
                matched_ids.append(document_id)
                matched_in_text[document_id] = found
        print('Length of second filtering: ', len(matched_ids))
        docs_features = WikiPageIndex.get_features(query_terms, matched_ids)
        docs_relevance = self.scoring_model.predict(docs_features)
        docs_relevance = list(docs_relevance[:, 1])

        matched_docs = []
        for score, document_id in sorted(zip(docs_relevance, matched_ids),
                                         key=lambda x: x[0], reverse=True)[:10]:
            document_line_number = TextIndex.start.select_value(document_id)
            xml = self.get_xml(self.dump_path, document_line_number)

            matched_docs.append(
                {
                    'title': self.get_title(xml),
                    'snippet': self.get_snippet(xml, matched_in_text[document_id]),
                    'href': 'http://www.{0}.com'.format(document_id),
                    'score': score
                })
        return matched_docs

    def get_title(self, xml):
        return xml.xpath("//title")[0].text

    def get_snippet(self, xml, matched):
        return TextIndex.tokenizer_stemmer.get_snippet(
            matched[0], matched[1], xml.xpath("//text")[0].text)


if __name__ == '__main__':
    DUMP_PATH = '../../enwiki-20171020-pages-articles1.xml-p10p30302'
    if 0:
        wiki_files = WikiIndex(DUMP_PATH).build_index_file()
        print('Index ready')
    else:
        wiki_files = WikiIndex(DUMP_PATH)
    result = wiki_files.search_query('world war II')
    print(result)
