import os
import numpy as np
import pickle
from catboost import CatBoostClassifier, cv, Pool


MSLR_PATH = '../../MSLR-WEB10K'
MSLR_DUMPS_PATH = '../../mslr_dumps'


def parse_file(g):
    features = []
    answer = []

    for line in g:
        line_splitted = line.strip().split(' ')
        answer.append(int(line_splitted[0]))

        features_vector = []
        for element in line_splitted[1:]:
            element_splitted = element.split(':')
            features_vector.append(float(element_splitted[1]))
        features.append(features_vector)
    return {'X': np.array(features, dtype=np.float32), 'y': np.asarray(answer, dtype=np.int8)}


def get_mslr_data():
    for fold in os.listdir(MSLR_PATH):
        print(fold)
        cur_fold = {}

        with open(os.path.join(MSLR_PATH, fold, 'train.txt')) as f:
            cur_fold['train'] = parse_file(f)
        print('train')
        with open(os.path.join(MSLR_PATH, fold, 'test.txt')) as f:
            cur_fold['test'] = parse_file(f)
        print('test')
        with open(os.path.join(MSLR_PATH, fold, 'vali.txt')) as f:
            cur_fold['vali'] = parse_file(f)
        print('vali')
        yield fold, cur_fold


def init_folds():
    for fold_name, data in get_mslr_data():
        with open(os.path.join(MSLR_DUMPS_PATH, fold_name), 'wb') as f:
            pickle.dump(data, f)
            return


def load_folds():
    for fold_name in os.listdir(MSLR_DUMPS_PATH):
        with open(os.path.join(MSLR_DUMPS_PATH, fold_name), 'rb') as f:
            yield fold_name, pickle.load(f)


def filter_features(np_array):
    exclude = [0] + list(range(101, 126)) + [128] + list(range(131, 137))
    return np.delete(np_array, exclude, axis=1)


def train_model():
    fold_name, fold = next(load_folds())
    train = fold['train']
    eval_set = fold['vali']

    cat_features = [96, 97, 98, 99, 100]
    model = CatBoostClassifier(custom_loss=['Accuracy'])
    print('Started training')

    model.fit(
        train['X'], train['y'],
        cat_features=cat_features,
        eval_set=(eval_set['X'], eval_set['y']),
        verbose=True,  # you can uncomment this for text output
        plot=True
    )

    # cv_data = cv(
    #     model.get_params(),
    #     Pool(train['X'], label=train['y'], cat_features=cat_features),
    # )
    #
    # print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    #     np.max(cv_data['Accuracy_test_avg']),
    #     cv_data['Accuracy_test_stddev'][np.argmax(cv_data['Accuracy_test_avg'])],
    #     np.argmax(cv_data['Accuracy_test_avg'])))


if __name__ == '__main__':
    train_model()
