#!/usr/bin/env python3
"""

"""

import numpy as np
import sklearn.metrics as metrics


class ASRSReportClassifier:

    def __init__(self, classifier, normalize_counts: bool = True):
        # TODO: Initialize an empty model for each field classifier
        self._model = None  # Not trained model yet
        self._classifier = classifier
        self._normalize = normalize_counts
        ex_dict = {
            0: 'Person Flight Crew',
            1: 'Person Air Traffic Control'
        }
        self._encoding = dict()  # int: label dictionary of possible classes

    def train_report_classification(self, train_data, train_target):

        print(f'Train data shape: {train_data.shape}')

        print(self._classifier)
        self._model = self._classifier.fit(train_data, train_target)

    def evaluate(self, test_data, test_target):
        predictions = self.predict(test_data)

        print('==============================================')
        print(f'Model Based Accuracy: {metrics.accuracy_score(test_target, predictions) * 100}')
        print(f'Model Based F1-score: {metrics.f1_score(test_target, predictions) * 100}')
        print('==============================================')
        predictions = np.zeros(test_target.shape)
        print(f'Accuracy predicting always 0: {metrics.accuracy_score(test_target, predictions) * 100}')
        print(f'F1-score: {metrics.f1_score(test_target, predictions) * 100}')
        print('==============================================')
        predictions = np.ones(test_target.shape)
        print(f'Accuracy predicting always 1: {metrics.accuracy_score(test_target, predictions) * 100}')
        print(f'F1-score: {metrics.f1_score(test_target, predictions) * 100}')

    def predict(self, test_data):
        if self._model is None:
            raise TypeError("A model needs to be trained first to perform predictions.")

        predictions = self._model.predict(test_data)

        return predictions

    def _decode_prediction(self, prediction: int):
        if not len(self._encoding):
            raise ValueError('Train a model to get an non-empty encoding.')

        decoded_label = self._encoding.get(prediction)

        if decoded_label is None:
            raise ValueError(f'Encoding with value "{prediction}" does not exist.')

        return decoded_label

    def decode_predictions(self, predictions: list):
        if predictions is None:
            raise TypeError('Predictions have to be made')

        vectorized = np.vectorize(self._decode_prediction)
        decoded_labels = vectorized(predictions)

        return decoded_labels

    def label_text(self, text):
        """

        :param text:
        :return:
        """
        # TODO: For a given text, the classifier returns a dictionary containing field name as key and its predicted value
        return 0