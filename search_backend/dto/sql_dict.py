from collections import defaultdict

from pymongo import MongoClient, IndexModel, ASCENDING, TEXT
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError


# TODO: try with sql instead of mongodb

class WordCount:
    def __init__(self, name, bulk_limit=1000000):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client.websearch
        self.cl = self.db['word_count_{0}'.format(name)]
        self.bulk_write = []
        self.bulk_limit = bulk_limit

    def drop(self):
        self.cl.drop()
        self.cl.create_index([("document", ASCENDING), ("word", 1)], unique=True)

    def execute_bulk(self):
        if len(self.bulk_write) > 0:
            try:
                self.cl.insert_many(self.bulk_write, ordered=False)
            except BulkWriteError as e:
                print(e.details)
                raise

        self.bulk_write = []

    def check_bulk(self):
        if len(self.bulk_write) > self.bulk_limit:
            self.execute_bulk()

    def set(self, document, word, count):
        self.check_bulk()
        self.bulk_write.append({'word': word, 'document': document, 'count': count})

    def insert(self, document, word_dict):
        self.check_bulk()
        for word in word_dict:
            self.bulk_write.append({'word': word, 'document': document, 'count': word_dict[word]})

    def select_count(self, document, word):
        found = self.cl.find_one({'word': word, 'document': document})
        return found['count'] if found else None

    def select_words(self, doc):
        return [elem['word'] for elem in self.cl.find({'document': doc})]

    def select(self):
        return self.cl.find()

    def doc_amount(self):
        return len(self.cl.distinct('document'))


class WordOccurrences:
    def __init__(self, name, bulk_limit=1000000):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client.websearch
        self.cl = self.db['word_occurrences_{0}'.format(name)]
        self.bulk_write = []
        self.bulk_limit = bulk_limit

    def drop(self):
        self.cl.drop()
        self.cl.create_index([("word", 1), ("document", ASCENDING)])

    def execute_bulk(self):
        if len(self.bulk_write) > 0:
            try:
                self.cl.insert_many(self.bulk_write, ordered=False)
            except BulkWriteError as e:
                print(e.details)
                raise
        self.bulk_write = []

    def check_bulk(self):
        if len(self.bulk_write) > self.bulk_limit:
            self.execute_bulk()

    def set(self, document, word, position):
        self.check_bulk()
        self.bulk_write.append({'word': word, 'document': document, 'position': position})

    def insert(self, document, word_dict):
        self.check_bulk()
        for word, position in word_dict:
            self.bulk_write.append({'word': word, 'document': document, 'position': position})

    def select_positions(self, document, word):
        found = self.cl.find({'word': word, 'document': document})
        for res in found:
            yield res['position']

    def select_documents(self, word):
        return self.cl.find({'word': word}).distinct('document')

    def get_words(self):
        return self.cl.distinct('word')

    def get_df(self):
        result = self.cl.aggregate([
            {'$group': {'_id': '$word', 'df1': {'$addToSet': '$document'}}},
            {'$project': {'df': {'$size': "$df1"}}}
        ], allowDiskUse=True)
        return result


class SimpleDict:
    def __init__(self, name, bulk_limit=1000):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client.websearch
        self.cl = self.db[name]
        self.bulk_write = []
        self.bulk_update = defaultdict(int)
        self.bulk_limit = bulk_limit

    def drop(self):
        self.cl.drop()
        self.cl.create_index([("key", 1)], unique=True)

    def execute_bulk(self):
        if len(self.bulk_write) > 0:
            try:
                self.cl.insert_many(self.bulk_write, ordered=False)
            except BulkWriteError as e:
                print(e.details)
                raise

        update = []
        for key, value in self.bulk_update.items():
            update.append(UpdateOne({'key': key},
                                    {'$inc': {'value': value}}, upsert=True))
        if len(update) > 0:
            self.cl.bulk_write(update, ordered=False)
        self.bulk_write = []
        self.bulk_update = defaultdict(int)

    def check_bulk(self):
        if len(self.bulk_write) > self.bulk_limit:
            self.execute_bulk()

    def set(self, key, value):
        self.check_bulk()
        self.bulk_write.append({'key': key, 'value': value})

    def increment(self, key, value):
        self.check_bulk()
        self.bulk_update[key] += value

    def select_value(self, key):
        found = self.cl.find_one({'key': key})
        return found['value'] if found else None

    def select_keys(self):
        return self.cl.distinct('key')

    def select(self):
        return self.cl.find()
