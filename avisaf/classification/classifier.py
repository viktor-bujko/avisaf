#!/usr/bin/env python3
"""

"""

import os
import lzma
import json
import pickle
import logging
import numpy as np
from re import sub
from datetime import datetime
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


class ASRSReportClassificationPredictor:

    def __init__(self, model=None, labels_encoding: dict = None, vectorizer=None, normalized: bool = True,
                 model_params: dict = None, trained_label: str = None, trained_filter: list = None, deviation_rate: float = 0.0):

        self._model = model  # Model(s) to be used for evaluation
        if model_params is not None:
            for param, value in model_params.items():
                setattr(self._model, param, value)
        self._encoding = dict() if labels_encoding is None else labels_encoding  # "int: label" dictionary of possible classes
        self._normalize = normalized
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer)
        self._vectorizer = vectorizer
        self._trained_label = trained_label
        self._trained_filter = trained_filter
        self._deviation_rate = deviation_rate

    def predict_report_class(self, texts_paths: list, label_to_test: str = None, label_filter: list = None):

        if label_to_test is None and self._trained_label is not None:
            label_to_test = self._trained_label

        if label_filter is None and self._trained_filter is not None:
            label_filter = self._trained_filter

        test_data, test_target = self._preprocessor.vectorize_texts(
            texts_paths,
            label_to_test,
            train=False,
            label_values_filter=label_filter
        )

        logging.debug(self._preprocessor.get_data_distribution(test_target)[1])

        if self._normalize:
            test_data, test_target, _, _ = self._preprocessor.normalize(test_data, test_target, self._deviation_rate)

        logging.debug(f'Test data shape: {test_data.shape}')
        predictions = self.predict_proba(test_data)
        return predictions, test_target

    def predict(self, test_data, model_to_use=None):
        model = self._model if model_to_use is None else model_to_use

        if model is None:
            msg = 'A model needs to be trained or loaded first to perform predictions.'
            logging.error(msg)
            raise ValueError(msg)
        logging.debug(f'Predictions made using model: {model}')
        predictions = model.predict(test_data)

        return predictions

    def predict_proba(self, test_data, model_to_use=None):
        model = self._model if model_to_use is None else model_to_use

        if model is None:
            msg = 'A model needs to be trained or loaded first to perform predictions.'
            logging.error(msg)
            raise ValueError(msg)
        logging.debug(f'Probability predictions made using model: {model}')
        predictions = model.predict_proba(test_data)

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


class ASRSReportClassificationEvaluator:

    def __init__(self):
        pass

    @staticmethod
    def evaluate(predictions: list, test_target):

        ensemble = ""
        if len(predictions) > 1:
            logging.debug(f"{len(predictions)} models ensembling")
            ensemble = f"(ensemble of {len(predictions)} models)"

        predictions = np.argmax(np.mean(predictions, axis=0), axis=1)

        unique_predictions_count = np.unique(test_target).shape[0]
        avg = 'binary' if unique_predictions_count == 2 else 'micro'

        print('==============================================')
        print('Confusion matrix: number [i,j] indicates the number of observations of class i which were predicted to be in class j')
        print(metrics.confusion_matrix(test_target, predictions))
        if ensemble:
            print(ensemble)
        print(f'Model Based Accuracy: {metrics.accuracy_score(test_target, predictions) * 100}')
        print(f'Model Based Precision: {metrics.precision_score(test_target, predictions) * 100}')
        print(f'Model Based Recall: {metrics.recall_score(test_target, predictions) * 100}')
        print(f'Model Based F1-score: {metrics.f1_score(test_target, predictions, average=avg) * 100}')
        print('==============================================')
        for unique_prediction in range(unique_predictions_count):
            predictions = np.full(test_target.shape, unique_prediction)
            print(f'Accuracy predicting always {unique_prediction}: {metrics.accuracy_score(test_target, predictions) * 100}')
            print(f'F1-score: {metrics.f1_score(test_target, predictions, average=avg) * 100}')
            print(f'Model Based Precision: {metrics.precision_score(test_target, predictions, zero_division=1) * 100}')
            print(f'Model Based Recall: {metrics.recall_score(test_target, predictions) * 100}')
            print('==============================================')

    @staticmethod
    def plot(probability_predictions, test_target):
        pass
        """preds = probability_predictions[:, 1]

        fpr, tpr, threshold = metrics.roc_curve(test_target, preds)
        roc_auc = metrics.auc(fpr, tpr)
        prec, recall, thr = metrics.precision_recall_curve(test_target, preds)

        plt.title('ROC Curve')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.plot(prec, recall)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()"""


class ASRSReportClassificationTrainer:

    def __init__(self, classifier: str = None, normalized: bool = True, vectorizer=None, deviation_rate: float = 0.0):
        # TODO: Initialize an empty model for each field classifier
        def set_classification_algorithm(classification_algorithm: str):
            available_classifiers = {
                'mlp': MLPClassifier(hidden_layer_sizes=32, alpha=0.001),
                'svm': SVC(probability=True),
                'tree': DecisionTreeClassifier(criterion='entropy', max_features=10000),
                'forest': RandomForestClassifier(n_estimators=150, criterion='entropy', min_samples_split=15),
                'knn': KNeighborsClassifier(n_neighbors=15),
            }

            # Setting a default classifier value
            _classifier = available_classifiers['knn']

            if available_classifiers.get(classification_algorithm) is not None:
                _classifier = available_classifiers[classification_algorithm]

            return _classifier

        self._classifier = set_classification_algorithm(classifier)
        self._encoding = dict()
        self._normalize = normalized
        self._model = None
        self._params = self._classifier.get_params()
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer)
        self._trained_texts = []
        self._trained_label = None
        self._filter = None
        self._deviation_rate = deviation_rate

    def train_report_classification(self, texts_paths: list, label_to_train: str, label_filter: list = None):

        self._trained_texts = texts_paths
        self._trained_label = label_to_train
        self._filter = label_filter

        train_data, train_target = self._preprocessor.vectorize_texts(
            texts_paths,
            label_to_train,
            train=True,
            label_values_filter=label_filter
        )

        # encoding is available only after texts vectorization
        self._encoding = self._preprocessor.get_encoding()
        logging.debug(self._preprocessor.get_data_distribution(train_target)[1])

        if self._normalize:
            train_data, train_target, filtered_data, filtered_targets = self._preprocessor.normalize(train_data, train_target, self._deviation_rate)
            logging.debug(f'Train data shape: {filtered_data.shape}')
            model = self._classifier.fit(filtered_data, filtered_targets)
            self.save_model(model)

        logging.debug(f'Train data shape: {train_data.shape}')
        logging.debug(self._classifier)

        self._model = self._classifier.fit(train_data, train_target)

        self.save_model(self._model)

        train_data_evaluator = ASRSReportClassificationPredictor(
            model=self._model,
            labels_encoding=self._encoding,
            vectorizer=self._preprocessor.vectorizer,
            trained_label=label_to_train,
            trained_filter=label_filter,
            normalized=self._normalize,
            model_params=self._model.get_params()
        )
        predictions = train_data_evaluator.predict_proba(train_data)
        ASRSReportClassificationEvaluator.evaluate([predictions], train_target)

    def save_model(self, model_to_save):
        model_dir_name = "asrs_classifier-{}-{}".format(
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            ",".join(("{}_{}".format(sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(self._params.items())))
        )

        model_dir_name = model_dir_name[:100]

        if self._normalize:
            model_dir_name += ',norm'

        import pathlib
        pathlib.Path("classifiers").mkdir(exist_ok=True)
        pathlib.Path("classifiers", model_dir_name).mkdir(exist_ok=False)

        with lzma.open(os.path.join("classifiers", model_dir_name, 'classifier.model'), 'wb') as model_file:
            logging.debug(f'Saving model: {model_to_save}')
            logging.debug(f'Saving vectorizer: {self._preprocessor.vectorizer}')
            pickle.dump((model_to_save, self._preprocessor.vectorizer), model_file)

        with open(os.path.join("classifiers", model_dir_name, 'parameters.json'), 'w', encoding="utf-8") as params_file:
            logging.debug(f'Saving parameters [encoding, model parameters, train_texts_paths, trained_label, label_filter]')
            parameters = {
                "encoding": self._encoding,
                "model_params": self._params,
                "trained_label": {self._trained_label: self._filter},
                "trained_texts": self._trained_texts
            }
            json.dump(parameters, params_file, indent=4)


def launch_classification(models_dir_paths: list, texts_paths: list, label: str, label_filter: list, algorithm: str, normalize: bool, train: bool, plot: bool):

    deviation_rate = np.random.uniform(low=0.95, high=1.05, size=None) if normalize else None  # 5% of maximum deviation between classes
    if train:
        logging.debug('Training')
        classifier = ASRSReportClassificationTrainer(
            classifier=algorithm,
            vectorizer=None,
            normalized=normalize,
            deviation_rate=deviation_rate
        )
        classifier.train_report_classification(texts_paths, label, label_filter)
    else:
        logging.debug('Testing')

        if models_dir_paths is None:
            raise ValueError("The path to the model cannot be null for testing")

        models_predictions = []
        test_targets = None
        for model_dir_path in models_dir_paths:

            with lzma.open(os.path.join(model_dir_path, 'classifier.model'), 'rb') as model_file:
                model, vectorizer = pickle.load(model_file)

            with open(os.path.join(model_dir_path, 'parameters.json'), 'r') as params_file:
                parameters = json.load(params_file)

            trained_label = list(parameters["trained_label"].keys())[0]
            trained_filter = parameters["trained_label"][trained_label]

            predictor = ASRSReportClassificationPredictor(
                model=model,
                model_params=parameters["model_params"],
                labels_encoding=parameters["encoding"],
                vectorizer=vectorizer,
                trained_label=trained_label,
                trained_filter=trained_filter,
                normalized=normalize,
                deviation_rate=deviation_rate
            )

            predictions, targets = predictor.predict_report_class(texts_paths, label, label_filter)
            models_predictions.append(predictions)
            if plot:
                ASRSReportClassificationEvaluator.plot(predictions, targets)
            if test_targets is None:
                test_targets = targets

        ASRSReportClassificationEvaluator.evaluate(models_predictions, test_targets)

        """
        pre = ASRSReportDataPreprocessor()

        _, targets = pre.vectorize_texts(texts_paths, label, train=True, label_values_filter=label_filter)
        d = pre.get_data_distribution(targets)
        print(d)
        print(f'Different values: {len(d[0])}')
        print(f'Percentages: {np.around(d[1] * 100, decimals=2)}')
        print(f'Elements: {np.sum(d[0])}')"""
