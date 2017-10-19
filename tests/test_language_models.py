import unittest

from search_backend.language_models import NGramModel


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
