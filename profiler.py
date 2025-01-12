import pandas as pd

from carte_ai.fasttext import fasttext_disk
from carte_ai.src.carte_table_to_graph import Table2GraphTransformer

from carte_ai.data.load_data import set_split

import line_profiler

from sklearn.preprocessing import LabelEncoder
import argparse
import time


def read_lung_cancer(path='lung_cancer_data.csv', encode_target=True, drop_bad_cols=True):
    data = pd.read_csv(path)
    if encode_target:
        # Targets need to be categorical, so let's convert here.
        encoder = LabelEncoder()
        data["Treatment"] = encoder.fit_transform(data["Treatment"])
    if drop_bad_cols:
        data = data.drop(columns=["Patient_ID"])
    return data


def lung_cancer(data=None, num_train=128, random_state=42):
    if data is None:
        data = read_lung_cancer()
    config_data = {"target_name": "Treatment"}
    return set_split(data, config_data, num_train, random_state)


@line_profiler.profile
def preprocess_data_inmemory(X_train, X_test, y_train, y_test):
    model_path = './cc.en.300.bin'
    preprocessor = Table2GraphTransformer(fasttext_model_path=model_path)
    X_train = preprocessor.fit_transform(X_train, y=y_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test, preprocessor


def preprocess_data_disk(X_train, X_test, y_train, y_test):
    ft_disk = fasttext_disk.FastTextOnDisk(lmdb_path='./fasttext_lmdb')
    preprocessor = Table2GraphTransformer(lm_model=ft_disk)
    X_train = preprocessor.fit_transform(X_train, y=y_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test, preprocessor


def preprocess(preprocessor, X):
    return preprocessor.transform(X)


@line_profiler.profile
def infer(estimator, X):
    y = estimator.predict_proba(X)
    return y


def main():
    parser = argparse.ArgumentParser(description='Process lung cancer data.')
    parser.add_argument('--ondisk', action='store_true', default=False, help='Use disk-based preprocessing')
    args = parser.parse_args()

    data = read_lung_cancer()
    random_state = 42
    num_train = int(data.shape[0] * 0.8)

    X_train, X_test, y_train, y_test = lung_cancer(data, num_train, random_state)
    print("Lung Cancer dataset:", X_train.shape, X_test.shape)

    print('Starting preprocessing...')
    if args.ondisk:
        _, _, _, _, preprocessor = preprocess_data_disk(X_train, X_test, y_train, y_test)
    else:
        _, _, _, _, preprocessor = preprocess_data_inmemory(X_train, X_test, y_train, y_test)

    durations = []
    for _ in range(50):
        random_row = X_train.sample(1)
        start = time.time()
        preprocess(preprocessor, random_row)
        durations.append(time.time() - start)

    print(f'Average time for preprocessing: {sum(durations) / len(durations)}')

    print('Finished preprocessing')


if __name__ == '__main__':
    main()
