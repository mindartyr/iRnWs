import unittest

from search_backend.language_models import NGramModel, NGramWordsModel


class TestNGram(unittest.TestCase):

    def test_3gram(self):
        rus_3gram = NGramModel(3, 'ru')
        ukr_3gram = NGramModel(3, 'uk')

        rus_3gram.process_dir('russian_books', "cp1251")
        ukr_3gram.process_dir('ukr_books', "utf-8")

        test_strings = [('Доброго вечора', 'uk'),
                        ('Доброго вечера', 'ru'),
                        ('На добраніч ', 'uk'),
                        ('Доброй ночи', 'ru'),
                        ('Щасти вам', 'uk')]

        for test_string, language in test_strings:

            estimated_language = NGramModel.get_language_of_a_query(test_string, [rus_3gram, ukr_3gram])
            assert language == estimated_language, (test_string, language, estimated_language)

    def test_word_gram(self):
        dost_word_gram = NGramWordsModel(3, 'dost')
        tolst_word_gram = NGramWordsModel(3, 'tolst')

        dost_word_gram.process_dir('russian_books/dostoevskii', "cp1251")
        tolst_word_gram.process_dir('russian_books/tolstoi', "cp1251")

        test_strings = [('Не то чтоб он был так труслив и забит, совсем даже напротив; но', 'dost'),
                        ('С вечера, на последнем переходе, был получен приказ, что главнокомандующий '
                         'будет смотреть полк на походе.', 'tolst')]

        for test_string, language in test_strings:
            estimated_language = NGramWordsModel.get_language_of_a_query(test_string, [dost_word_gram, tolst_word_gram])

            assert language == estimated_language, (test_string, language, estimated_language)
