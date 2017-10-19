from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


class TokenizerStemmer:
    def __init__(self):
        self.eng_stop_words = set(stopwords.words('english'))
        self.eng_stemmer = SnowballStemmer("english")
        self.tokenizer = RegexpTokenizer(r'\w+')

    def tokenize_and_stem(self, text):
        for start, end in self.tokenizer.span_tokenize(text):
            word = text[start:end].lower()
            stemmed_word = self.eng_stemmer.stem(word)
            if word not in self.eng_stop_words:
                yield start, end, stemmed_word
