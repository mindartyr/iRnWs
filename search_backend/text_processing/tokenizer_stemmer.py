import os
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
from nltk.tokenize import sent_tokenize


class TokenizerStemmer:
    def __init__(self):
        self.eng_stop_words = set(stopwords.words('english'))
        self.eng_stemmer = SnowballStemmer("english")
        self.tokenizer = RegexpTokenizer(r'\w+')

    def span_tokenize(self, text):
        for start, end in self.tokenizer.span_tokenize(text):
            word = text[start:end].lower()
            stemmed_word = self.eng_stemmer.stem(word)
            stemmed_word = word
            if word not in self.eng_stop_words:
                yield start, end, stemmed_word

    def tokenize(self, text):
        text = text.lower()
        return self.tokenizer.tokenize(text)


class Tokenizer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')

    def span_tokenize(self, text):
        for start, end in self.tokenizer.span_tokenize(text):
            word = text[start:end].lower()
            yield start, end, word

    def tokenize(self, text):
        text = text.lower()
        return self.tokenizer.tokenize(text)


def process_dir(root_path, encoding='utf-8'):
    for dirName, subdirList, fileList in os.walk(root_path):
        for file_name in fileList:
            if file_name.endswith('.txt'):
                with open(os.path.join(dirName, file_name), 'r',  encoding=encoding) as f:
                    text = f.read()
                    yield text


class ToktokTokeniserStemmer:
    def __init__(self):
        self.eng_stemmer = PorterStemmer()
        self.tokenizer = ToktokTokenizer()
        self.stemmed_cache = {}

    def tokenize(self, text):
        for w_number, word in enumerate(self.tokenizer.tokenize(text)):
            word = word.lower().strip()
            stemmed_word = self.stemmed_cache.get(word, None)
            if not stemmed_word:
                if word != '':
                    stemmed_word = self.eng_stemmer.stem(word)
                else:
                    stemmed_word = ''
                self.stemmed_cache[word] = stemmed_word
            yield w_number, stemmed_word

    def get_snippet(self, beginning, end, text):
        snippet = []
        for w_number, word in enumerate(self.tokenizer.tokenize(text)):
            if w_number < beginning-10:
                continue
            else:
                snippet.append(word)
                if len(snippet) > end-beginning + 20:
                    return ' '.join(snippet)
        return ' '.join(snippet)
