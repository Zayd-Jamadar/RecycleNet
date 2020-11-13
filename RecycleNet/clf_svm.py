DATASET_DIR = './dataset'
MODEL_DIR = './trained_models/final_model.sav'

import numpy as np
from sklearn.svm import SVC
from feature_extraction import extract_features
import pickle

C = 1000
GAMMA = 0.5
sample_count=1504

def train_svm():
    svm_features, svm_labels = extract_features(DATASET_DIR,sample_count)
    X_train = svm_features.reshape(1504, 7*7*2048)
    y_train = svm_labels

    clf_svm = SVC(C=C, kernel='rbf', gamma=GAMMA)

    print('Now training the extracted features on SVM...')
    clf_svm.fit(X_train, y_train)

    pickle.dump(clf_svm, open(MODEL_DIR, 'rb'))

if __name__ == '__main__':
    train_svm()