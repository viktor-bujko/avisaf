#!/usr/bin/env python3
"""

"""

import lzma
import pickle
import logging
from datetime import datetime
import numpy as np
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from avisaf.training.training_data_creator import ASRSReportDataPreprocessor

logging.basicConfig(
    level=logging.DEBUG,
    format=f'[%(levelname)s - %(asctime)s]: %(message)s'
)


class ASRSReportClassifier:

    def __init__(self, classifier: str = None, normalized_distribution: bool = True, model=None, labels_encoding: dict = None, vectorizer=None):
        # TODO: Initialize an empty model for each field classifier

        def set_classification_algorithm(classification_algorithm: str):
            available_classifiers = {
                'mlp': (
                    MLPClassifier(hidden_layer_sizes=32, alpha=0.001),
                    "mlp-hidden-32,alpha-0_001"
                ),
                'svm': (SVC(), "svc_default"),
                'tree': (
                    DecisionTreeClassifier(criterion='entropy', max_features=5000),
                    "svm-crit-entropy,max_feat-5000"
                ),
                'forest': (
                    RandomForestClassifier(n_estimators=150, criterion='entropy', min_samples_split=15),
                    "forest-estimators-150,crit-5000,min_samples-15"
                ),
                'knn': (KNeighborsClassifier(n_neighbors=15), "knn-neigh-15")
            }
            # Setting a default classifier value
            _classifier, params = available_classifiers['mlp']

            if available_classifiers.get(classification_algorithm) is not None:
                _classifier, params = available_classifiers.get(classification_algorithm)
                self._classifier = _classifier
                self._params = params

            return _classifier, params

        self._model = model  # Not trained model yet
        self._classifier, self._params = set_classification_algorithm(classifier)
        self._normalize = normalized_distribution
        self._encoding = dict() if labels_encoding is None else labels_encoding  # "int: label" dictionary of possible classes
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer)

    def train_report_classification(self, texts_paths: list, label_to_train: str, label_filter: list = None):

        train_data, train_target = self._preprocessor.vectorize_texts(
            texts_paths,
            label_to_train,
            train=True,
            label_values_filter=label_filter
        )

        self._encoding = self._preprocessor.get_encoding()
        logging.debug(self._preprocessor.get_data_distribution(train_target)[1])

        if self._normalize:
            self._params += ',normalized'
            train_data, train_target = self._preprocessor.normalize(train_data, train_target)

        logging.debug(f'Train data shape: {train_data.shape}')
        logging.debug(self._classifier)

        self._model = self._classifier.fit(train_data, train_target)

        model_file_name = "report_classification_{}-{}.model".format(
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            self._params
        )

        with lzma.open(model_file_name, 'wb') as model_file:
            logging.debug(f'Saving model: {self._model}')
            logging.debug(f'Saving encoding: {self._encoding}')
            logging.debug(f'Saving vectorizer: {self._preprocessor.vectorizer}')
            pickle.dump((self._model, self._encoding, self._preprocessor.vectorizer), model_file)
        self.evaluate(train_data, train_target)

    def evaluate_report_classification(self, texts_paths: list, label_to_test: str, label_filter: list = None):

        test_data, test_target = self._preprocessor.vectorize_texts(
            texts_paths,
            label_to_test,
            train=False,
            label_values_filter=label_filter
        )

        logging.debug(self._preprocessor.get_data_distribution(test_target)[1])

        if self._normalize:
            test_data, test_target = self._preprocessor.normalize(test_data, test_target)

        logging.debug(f'Test data shape: {test_data.shape}')
        self.evaluate(test_data, test_target)

    def evaluate(self, test_data, test_target):
        predictions = self.predict(test_data)

        unique_predictions_count = np.unique(predictions).shape[0]
        avg = 'binary' if unique_predictions_count == 2 else 'micro'

        print('==============================================')
        print(f'Model Based Accuracy: {metrics.accuracy_score(test_target, predictions) * 100}')
        print(f'Model Based F1-score: {metrics.f1_score(test_target, predictions, average=avg) * 100}')
        print('==============================================')
        for unique_prediction in range(unique_predictions_count):
            predictions = np.full(test_target.shape, unique_prediction)
            print(f'Accuracy predicting always {unique_prediction}: {metrics.accuracy_score(test_target, predictions) * 100}')
            print(f'F1-score: {metrics.f1_score(test_target, predictions, average=avg) * 100}')
            print('==============================================')

    def predict(self, test_data):
        if self._model is None:
            msg = 'A model needs to be trained or loaded first to perform predictions.'
            logging.error(msg)
            raise ValueError(msg)
        logging.debug(f'Predictions made using model: {self._model}')
        predictions = self._model.predict(test_data)

        return predictions

    def _decode_prediction(self, prediction: int):
        if not len(self._encoding):
            msg = 'Train a model to get an non-empty encoding.'
            logging.error(msg)
            raise ValueError(msg)

        decoded_label = self._encoding.get(prediction)

        if decoded_label is None:
            msg = f'Encoding with value "{prediction}" does not exist.'
            logging.error(msg)
            raise ValueError(msg)

        return decoded_label

    def decode_predictions(self, predictions: list):
        if predictions is None:
            msg = 'Predictions have to be made first'
            logging.error(msg)
            raise TypeError(msg)

        vectorized = np.vectorize(self._decode_prediction)
        decoded_labels = vectorized(predictions)

        return decoded_labels

    def label_text(self, text):
        """

        :param text:
        :return:
        """
        if self._vectorizer is None:
            msg = 'A model needs to be trained or loaded first to be able to transform texts.'
            logging.error(msg)
            raise ValueError(msg)

        vectorized_text = self._vectorizer.transform(text)
        prediction = self._model.predict(vectorized_text)
        predicted_label = self._decode_prediction(prediction)

        # TODO: For a given text, the classifier returns a dictionary containing field name as key and its predicted value
        return predicted_label


def launch_classification(model_path: str, texts_paths: list, label: str, label_filter: list, algorithm: str, normalize: bool, test: bool, train: bool):

    if test:
        logging.debug('Testing')

        if model_path is None:
            raise ValueError("The path to the model cannot be null for testing")

        with lzma.open(model_path, 'rb') as model_file:
            model, encoding, vectorizer = pickle.load(model_file)

        classifier = ASRSReportClassifier(
            model=model,
            labels_encoding=encoding,
            vectorizer=vectorizer,
            normalized_distribution=normalize
        )
        classifier.evaluate_report_classification(texts_paths, label, label_filter)
    elif train:
        logging.debug('Training')
        classifier = ASRSReportClassifier(
            classifier=algorithm,
            vectorizer=None,
            normalized_distribution=normalize
        )
        classifier.train_report_classification(texts_paths, label, label_filter)

        """
        pre = ASRSReportDataPreprocessor()

        _, targets = pre.convert_texts_to_matrices(texts_paths, label, train=True, label_values_filter=None)
        d = pre.get_data_distribution(targets)
        print(d)
        print(f'Different values: {len(d[0])}')
        print(f'Percentages: {d[1] * 100}')
        """
