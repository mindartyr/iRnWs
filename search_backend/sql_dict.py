from collections import defaultdict

from pymongo import MongoClient, IndexModel, ASCENDING, TEXT
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from pymongo.errors import BulkWriteError


# TODO: try with sql instead of mongodb

class WordCount:
    def __init__(self, name, bulk_limit=1000):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client.websearch
        self.cl = self.db['word_count_{0}'.format(name)]
        self.cl.drop()
        self.bulk_write = []
        self.bulk_update = defaultdict(int)
        self.bulk_limit = bulk_limit
        self.cl.create_index([("document", ASCENDING), ("word", 1)], unique=True)

    def execute_bulk(self):
        if len(self.bulk_write) > 0:
            try:
                self.cl.insert_many(self.bulk_write)
            except BulkWriteError as e:
                print(e.details)
                raise

        update = []
        for key, value in self.bulk_update.items():
            update.append(UpdateOne({'document': key[0], 'word': key[1]},
                                    {'$inc': {'count': value}}, upsert=True))
        if len(update) > 0:
            self.cl.bulk_write(update, ordered=False)
        self.bulk_write = []
        self.bulk_update = defaultdict(int)

    def check_bulk(self):
        if len(self.bulk_write) > self.bulk_limit:
            self.execute_bulk()

    def set(self, document, word, count):
        self.check_bulk()
        self.bulk_write.append({'word': word, 'document': document, 'count': count})

    def increment(self, document, word, count):
        self.check_bulk()
        self.bulk_update[(document, word)] += count

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
    def __init__(self, name, bulk_limit=1000):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client.websearch
        self.cl = self.db['word_occurrences_{0}'.format(name)]
        self.cl.drop()
        self.bulk_write = []
        self.bulk_update = defaultdict(list)
        self.bulk_limit = bulk_limit
        self.cl.create_index([("word", 1), ("document", ASCENDING)], unique=True)

    def execute_bulk(self):
        if len(self.bulk_write) > 0:
            try:
                self.cl.insert_many(self.bulk_write)
            except BulkWriteError as e:
                print(e.details)
                raise

        update = []
        for key, value in self.bulk_update.items():
            update.append(UpdateOne({'document': key[0], 'word': key[1]},
                                    {'$push': {'position': {'$each': value}}}, upsert=True))
        if len(update) > 0:
            self.cl.bulk_write(update, ordered=False)
        self.bulk_write = []
        self.bulk_update = defaultdict(list)

    def check_bulk(self):
        if len(self.bulk_write) > self.bulk_limit:
            self.execute_bulk()

    def set(self, document, word, position):
        self.check_bulk()
        self.bulk_write.append({'word': word, 'document': document, 'position': [position]})

    def append(self, document, word, position):
        self.check_bulk()
        self.bulk_update[(document, word)].append(position)

    def select_positions(self, document, word):
        found = self.cl.find_one({'word': word, 'document': document})
        return found['position'] if found else []

    def select_documents(self, word):
        return [elem['document'] for elem in self.cl.find({'word': word})]

    def select(self):
        return self.cl.find()

    def doc_amount(self):
        return len(self.cl.distinct('document'))


class SimpleDict:
    def __init__(self, name, bulk_limit=1000):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client.websearch
        self.cl = self.db[name]
        self.cl.drop()
        self.bulk_write = []
        self.bulk_update = defaultdict(int)
        self.bulk_limit = bulk_limit
        self.cl.create_index([("key", 1)], unique=True)

    def execute_bulk(self):
        if len(self.bulk_write) > 0:
            try:
                self.cl.insert_many(self.bulk_write)
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
