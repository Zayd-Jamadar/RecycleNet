DATASET_DIR = './dataset'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from feature_extraction import extract_features

C = 1000
GAMMA = 0.5
sample_count=1504

def train_svm():
    svm_features, svm_labels = extract_features(DATASET_DIR,sample_count)
    X_train = svm_features.reshape(1504, 7*7*2048)
    y_train = svm_labels

    clf_svm = SVC(C=C, kernel='rbf', gamma=GAMMA)

    clf_svm.fit(X_train, y_train)

if __name__ == '__main__':
    train_svm()