from collections import defaultdict

from search_backend.tokenizer_stemmer import process_dir, Tokenizer


class TextRank:

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.betta = 0.15
        self.stop_eps = 0.1

    def process_dir(self, root_path, encoding="utf-8"):
        for text in process_dir(root_path, encoding):
            best_snippet = self.process_text(text)
            print("Best snippet: ", best_snippet)
        return self

    def process_text(self, text):
        bags = dict()
        scores = defaultdict(dict)
        common_word_amount = defaultdict(dict)

        for p_number, paragraph in enumerate(self.get_paragraphs(text)):
            bags[p_number] = self.get_bag_of_words(paragraph)

        for p_number_1 in bags:
            words_cardinality = 0

            for p_number_2 in bags:
                if p_number_1 == p_number_2:
                    continue

                if p_number_2 in common_word_amount:
                    if p_number_1 in common_word_amount[p_number_2]:
                        common_word_amount[p_number_1][p_number_2] = \
                            common_word_amount[p_number_2][p_number_1]

                total_common_words_amount = 0
                for word in bags[p_number_1]:
                    if word in bags[p_number_2]:
                        total_common_words_amount += min(bags[p_number_1][word],
                                                         bags[p_number_2][word])
                if total_common_words_amount > 0:
                    common_word_amount[p_number_1][p_number_2] = total_common_words_amount
                    words_cardinality += total_common_words_amount

            for p_number_2 in common_word_amount[p_number_1]:
                # make probabilities less than 1
                if words_cardinality > 0:
                    scores[p_number_1][p_number_2] = \
                        common_word_amount[p_number_1][p_number_2] / words_cardinality

        best_paragraph_number = self.find_best_paragraph(scores)

        for p_number, paragraph in enumerate(self.get_paragraphs(text)):
            if p_number == best_paragraph_number:
                return paragraph

        return None

    def find_best_paragraph(self, scores):
        paragraph_rank = defaultdict(int)

        while 1:
            cumulative_change = 0

            for p_number_1 in scores:
                old_value = paragraph_rank[p_number_1]

                multiplier = 0
                for p_2 in scores[p_number_1]:
                    if p_2 == p_number_1:
                        continue
                    multiplier += paragraph_rank[p_2] * scores[p_number_1][p_2]

                paragraph_rank[p_number_1] = \
                    self.betta + \
                    (1 - self.betta) * multiplier / len(scores[p_number_1])
                cumulative_change += abs(old_value - paragraph_rank[p_number_1])

            if cumulative_change < self.stop_eps:
                break
        return max(paragraph_rank, key=lambda x: paragraph_rank[x])

    def get_bag_of_words(self, text):
        word_count = defaultdict(int)
        for word in self.tokenizer.tokenize(text):
            word_count[word] += 1
        return word_count

    @staticmethod
    def get_paragraphs(text):
        min_paragraph_len = 100
        cur_paragraph = ''

        for par in text.split('\n\n'):
            cur_paragraph += par
            if len(cur_paragraph) < min_paragraph_len:
                continue
            else:
                yield cur_paragraph
                cur_paragraph = ''


if __name__ == '__main__':
    TextRank().process_dir('../data/test_text/')
