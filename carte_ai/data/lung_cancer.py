from sklearn.preprocessing import LabelEncoder
from carte_ai.data.load_data import set_split
import pandas as pd


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
