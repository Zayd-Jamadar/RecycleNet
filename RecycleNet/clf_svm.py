import os
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_extraction import extract_features
import pickle

date_string = time.strftime("%Y-%m-%d-%H:%M")

DATASET_DIR = './dataset/train'
FILE_NAME = 'resnet_svm_' + date_string + '.sav'
MODEL_DIR = './trained_models/' + FILE_NAME

C = 1000
GAMMA = 0.5
sample_count = 1050

def plot_cf(y_pred, y_test):
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

def train_svm():
    print(FILE_NAME)
    svm_features, svm_labels = extract_features(DATASET_DIR,sample_count)
    X = svm_features.reshape(1504, 7*7*2048)
    y = svm_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf_svm = SVC(C=C, kernel='rbf', gamma=GAMMA)

    print("Fitting the model...")
    clf_svm.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    pickle.dump(clf_svm, open(MODEL_DIR, 'wb'))

    y_pred = clf_svm.predict(X_test)

    print("Model is trained. Plotting metrics...")
    plot_cf(y_pred, y_test)

if __name__ == '__main__':
    print('Now training the extracted features on SVM...')
    train_svm()