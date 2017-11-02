import os
import numpy as np
import pickle
from catboost import CatBoostClassifier, cv, Pool
from sklearn.preprocessing import MinMaxScaler


MSLR_PATH = '../../../MSLR-WEB10K'
MSLR_DUMPS_PATH = '../../../mslr_dumps'
DIR = os.path.dirname(os.path.realpath(__file__))


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


def get_features_mapping(amount):
    output = []
    i = 1
    while len(output) < amount:
        if i % 5 != 0 and i % 5 != 4:
            output.append(i)
        i += 1
    return output


def load_folds():
    for fold_name in os.listdir(MSLR_DUMPS_PATH):
        with open(os.path.join(MSLR_DUMPS_PATH, fold_name), 'rb') as f:
            yield fold_name, pickle.load(f)


def filter_features(np_array):
    include = get_features_mapping(19 * 3)
    include.remove(16)
    include.remove(17)
    include.remove(18)
    return MinMaxScaler().fit_transform(np_array[:, include])


def modify_answers(np_array):
    np_array[np_array == 1] = 0
    np_array[np_array == 2] = 1

    np_array[np_array == 3] = 1
    np_array[np_array == 4] = 1

    return np_array


def oversample(X, y):
    repeat_factor = int(len(y[y==0]) / len(y[y==1]))
    new_X = np.concatenate((X[y==0], X[y==1].repeat(repeat_factor, axis=0)), axis=0)
    new_y = np.concatenate((y[y==0], y[y==1].repeat(repeat_factor, axis=0)), axis=0)
    return new_X, new_y


class ScoringModel:
    def __init__(self):
        self.model = None

    def load_model(self):
        with open(os.path.join(DIR, 'catboost_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        return self

    def predict(self, X, scale=True):
        if scale:
            X = MinMaxScaler().fit_transform(X)
        return self.model.predict_proba(X)

