#!/usr/bin/env python3
"""

"""

import lzma
import json
import pickle
import logging
import numpy as np
from re import sub
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import cross_validate, cross_val_score, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC

from avisaf.training.training_data_creator import ASRSReportDataPreprocessor
logger = logging.getLogger(str(__file__))

logging.basicConfig(
    level=logging.DEBUG,
    format=f'[%(levelname)s - %(asctime)s]: %(message)s'
)


class ASRSReportClassificationPredictor:

    def __init__(self, models, vectorizer=None, normalized: bool = True, deviation_rate: float = 0.0, parameters=None):

        if parameters is None:
            parameters = {}
        self._models = models  # Model(s) to be used for evaluation

        if parameters.get("model_params") is not None:
            for model in self._models.values():
                for param, value in parameters["model_params"].items():
                    setattr(model, param, value)

        self._normalize = normalized
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer)
        self._vectorizer = vectorizer
        self._deviation_rate = deviation_rate

        try:
            self._encodings = parameters["encodings"]  # "int: label" dictionary of possible classes
            self._trained_filtered_labels = parameters["trained_labels"]
        except AttributeError:
            raise ValueError("Corrupted model parameters")

        assert len(self._models.keys()) == len(self._trained_filtered_labels.keys())

    """def predict_report_class(self, texts_paths: list, label_to_test: str = None, label_filter: list = None):
        
        labels_to_test = self._trained_filtered_labels.keys()
        labels_filters = self._trained_filtered_labels.values()

        if label_to_test is not None:
            labels_to_test = [label_to_test]
            
            if label_filter is None:
                if self._trained_filtered_labels.get(label_to_test):
                    labels_filters = self._trained_filtered_labels[label_to_test]
                else:
                    labels_filters = None
            else:
                labels_filters = [label_filter]
            
        for lbl, fltr in zip(labels_to_test, labels_filters):

            test_data, test_target = self._preprocessor.vectorize_texts(
                texts_paths,
                lbl,
                train=False,
                label_values_filter=fltr,
                normalize=self._normalize
            )"""

    @staticmethod
    def predict(test_data, model=None):

        if model is None:
            raise ValueError('A model needs to be trained or loaded first to perform predictions.')

        logger.info(f'Test data shape: {test_data.shape}')
        predictions = model.predict(test_data, None)

        return predictions

    @staticmethod
    def predict_proba(test_data, model=None):
        if model is None:
            raise ValueError('A model needs to be trained or loaded first to perform predictions.')

        logger.info(f'Probability predictions made using model: {model}')
        if getattr(model, 'predict_proba', None) is not None:
            predictions = model.predict_proba(test_data)
        else:
            predictions = model.predict(test_data)

        return predictions

    def _decode_prediction(self, prediction: int):
        if not len(self._encodings):
            raise ValueError('Train a model to get an non-empty encoding.')

        decoded_label = self._encodings.get(prediction)

        if decoded_label is None:
            raise ValueError(f'Encoding with value "{prediction}" does not exist.')

        return decoded_label

    def decode_predictions(self, predictions: list):
        if predictions is None:
            raise TypeError('Predictions have to be made first')

        vectorized = np.vectorize(self._decode_prediction)
        decoded_labels = vectorized(predictions)

        return decoded_labels

    def label_text(self, text):
        """

        :param text:
        :return:
        """
        if self._vectorizer is None:
            raise ValueError('A model needs to be trained or loaded first to be able to transform texts.')

        vectorized_text = self._vectorizer.transform(text)
        prediction = self._models.predict(vectorized_text)
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
            logger.debug(f"{len(predictions)} models ensembling")
            ensemble = f"(ensemble of {len(predictions)} models)"

        predictions = np.mean(predictions, axis=0)

        for predictions_distribution, class_targets in zip(predictions, test_target):
            class_predictions = np.argmax(predictions_distribution, axis=1)
            unique_predictions_count = np.unique(class_targets).shape[0]
            avg = 'binary' if unique_predictions_count == 2 else 'macro'

            print('==============================================')
            print('Confusion matrix: number [i,j] indicates the number of observations of class i which were predicted to be in class j')
            print(metrics.confusion_matrix(class_targets, class_predictions))
            if ensemble:
                print(ensemble)
            print('Model Based Accuracy: {:.2f}'.format(metrics.accuracy_score(class_targets, class_predictions) * 100))
            print('Model Based Balanced Accuracy: {:.2f}'.format(metrics.balanced_accuracy_score(class_targets, class_predictions) * 100))
            print('Model Based ROC-AUC: {:.2f}'.format(metrics.roc_auc_score(class_targets, predictions_distribution, multi_class='ovr') * 100))

            print('Model Based Macro Precision: {:.2f}'.format(metrics.precision_score(class_targets, class_predictions, average=avg) * 100))
            print('Model Based Macro Recall: {:.2f}'.format(metrics.recall_score(class_targets, class_predictions, average=avg) * 100))
            print('Model Based Macro F1-score: {:.2f}'.format(metrics.f1_score(class_targets, class_predictions, average=avg) * 100))
            print('==============================================')
            for unique_prediction in range(unique_predictions_count):
                mockup_predictions = np.full(class_targets.shape, unique_prediction)
                print(f'Accuracy predicting always {unique_prediction}: {metrics.accuracy_score(class_targets, mockup_predictions) * 100}')
                print(f'Accuracy predicting always {unique_prediction}: {metrics.balanced_accuracy_score(class_targets, mockup_predictions) * 100}')
                print(f'F1-score: {metrics.f1_score(class_targets, mockup_predictions, average=avg) * 100}')
                print(f'Model Based Precision: {metrics.precision_score(class_targets, mockup_predictions, zero_division=1, average=avg) * 100}')
                print(f'Model Based Recall: {metrics.recall_score(class_targets, mockup_predictions, average=avg) * 100}')
                print('==============================================')

    @staticmethod
    def plot(probability_predictions, test_target):
        preds = np.mean(probability_predictions, axis=0)[:, 1]

        fpr, tpr, threshold = metrics.roc_curve(test_target, preds)
        roc_auc = metrics.auc(fpr, tpr)
        # prec, recall, thr = metrics.precision_recall_curve(test_target, preds)

        plt.title('ROC Curve')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    @staticmethod
    def plot_learning_curve(estimator, title, x, y, axes=None, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, x, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt


class ASRSReportClassificationTrainer:

    def __init__(self, models: dict = None, encoders: list = None, parameters: dict = None, algorithm=None, normalized: bool = True, deviation_rate: float = 0.0):

        def set_classification_algorithm(classification_algorithm: str):
            available_classifiers = {
                'mlp': MLPClassifier(
                    hidden_layer_sizes=(512, 256),
                    alpha=0.007,
                    batch_size=128,
                    learning_rate='adaptive',
                    learning_rate_init=0.003,
                    verbose=True,
                    early_stopping=True,
                    n_iter_no_change=20,
                ),
                'svm': LinearSVC(dual=False, class_weight='balanced'),
                'forest': RandomForestClassifier(
                    n_estimators=150,
                    criterion='entropy',
                    min_samples_split=32,
                    n_jobs=2,
                    verbose=5
                ),
                'knn': KNeighborsClassifier(n_neighbors=20, weights='distance'),
                'gauss': GaussianNB(),
                'mnb': MultinomialNB(),
                'regression': LogisticRegression()
            }

            # Setting a default classifier value
            _classifier = available_classifiers['knn']

            if available_classifiers.get(classification_algorithm) is not None:
                _classifier = available_classifiers[classification_algorithm]

            return _classifier

        if parameters is None:
            parameters = {}

        self._normalize = normalized
        self._preprocessor = ASRSReportDataPreprocessor(encoders=encoders)

        if not models:
            self._classifier = set_classification_algorithm(algorithm)
            self._models = {}
            self._deviation_rate = 0.0
            self._encodings = {}
            self._model_params = self._classifier.get_params()
            self._params = {}
            self._algorithm = algorithm
            self._trained_filtered_labels = {}
            self._trained_texts = []
        else:
            try:
                self._classifier = list(models.values())[0]
                self._models = models
                self._deviation_rate = deviation_rate
                self._params = parameters
                encodings = {}
                for label, encoding in parameters["encodings"].items():
                    encodings.update({label: {int(key): value for key, value in encoding.items()}})
                self._encodings = encodings
                self._model_params = parameters["model_params"]
                self._algorithm = parameters["algorithm"]
                self._trained_filtered_labels = parameters["trained_labels"]
                self._trained_texts = parameters["trained_texts"]
            except AttributeError:
                raise ValueError("Corrupted parameters.json file")

        assert self._models.keys() == self._encodings.keys()

        if self._classifier is not None and parameters.get("model_params") is not None:
            for param, value in parameters["model_params"].items():
                try:
                    setattr(self._classifier, param, value)
                except AttributeError:
                    logging.warning(f"Trying to set a non-existing attribute { param } with value { value }")

    def train_report_classification(self, texts_paths: list, label_to_train: str, label_filter: list = None, mode: str = "dev"):

        if mode not in ["train", "dev", "test"]:
            raise ValueError("Unsupported argument")

        if mode == "train":
            self._trained_texts += texts_paths

        labels_to_train = list(self._trained_filtered_labels.keys())
        labels_filters = list(self._trained_filtered_labels.values())

        if label_to_train is not None:
            labels_to_train = [label_to_train]

            if label_filter is None:
                # trying to get previously saved label filter
                if self._trained_filtered_labels.get(label_to_train):
                    labels_filters = self._trained_filtered_labels[label_to_train]
                    filter_update = labels_filters
                else:
                    labels_filters = None
                    filter_update = []
            else:
                labels_filters = [label_filter]
                filter_update = label_filter

            if mode == "train":
                self._trained_filtered_labels.update({label_to_train: filter_update})

        if not labels_to_train:
            raise ValueError("Nothing to train - please make sure at least one category is specified.")

        assert len(labels_to_train) == len(labels_filters)

        labels_predictions, labels_targets = [], []

        data, target = self._preprocessor.vectorize_texts(
            texts_paths,
            labels_to_train,
            train=mode == "train",
            label_values_filter=labels_filters,
            normalize=self._normalize
        )

        for i, zipped in enumerate(zip(labels_to_train, labels_filters)):

            lbl, fltr = zipped

            logger.debug(f'{ mode } data shape: {data[i].shape}')
            logger.debug(self._preprocessor.get_data_distribution(target[i])[1])

            if self._models.get(lbl) is not None:
                logging.debug("Found previously trained model")
                classifier = self._models[lbl]
                setattr(classifier, 'warm_start', True)
                setattr(classifier, 'learning_rate_init', 0.0005)
            else:
                classifier = clone(self._classifier)
            if mode == "train":
                # encoding is available only after texts vectorization
                encoding = {}
                for label_idx, label in enumerate(self._preprocessor.encoder(i).classes_):
                    encoding.update({label_idx: label})
                self._encodings.update({lbl: encoding})

                self._params = {
                    "algorithm": self._algorithm,
                    "encodings": self._encodings,
                    "model_params": self._model_params,
                    "trained_labels": self._trained_filtered_labels,
                    "trained_texts": self._trained_texts,
                    "vectorizer_params": self._preprocessor.vectorizer.get_params()
                }

                classifier.fit(data[i], target[i])
                logging.info(f"MODEL: {classifier}")
                self._models.update({lbl: classifier})

            predictions = ASRSReportClassificationPredictor.predict_proba(data[i], classifier)
            labels_predictions.append(predictions)
            labels_targets.append(target[i])
            if mode == "train":
                ASRSReportClassificationEvaluator.evaluate([[predictions]], [target[i]])

        if mode == "train":
            self.save_model(self._models)

        return labels_predictions, labels_targets

    def save_model(self, models_to_save: dict):
        model_dir_name = "asrs_classifier-{}-{}-{}".format(
            self._algorithm,
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            ",".join(("{}_{}".format(sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(self._model_params.items()))).replace(" ", "_", -1)
        )

        model_dir_name = model_dir_name[:100]

        if self._normalize:
            model_dir_name += ',norm'

        Path("classifiers").mkdir(exist_ok=True)
        Path("classifiers", model_dir_name).mkdir(exist_ok=False)
        with lzma.open(Path("classifiers", model_dir_name, 'classifier.model'), 'wb') as model_file:
            logger.info(f'Saving {len(models_to_save)} model(s): {models_to_save}')
            pickle.dump((models_to_save, self._preprocessor.encoders), model_file)

        with open(Path("classifiers", model_dir_name, 'parameters.json'), 'w', encoding="utf-8") as params_file:
            logger.info(f'Saving parameters [encoding, model parameters, train_texts_paths, trained_labels, label_filter]')
            json.dump(self._params, params_file, indent=4)


def launch_classification(models_dir_paths: list, texts_paths: list, label: str, label_filter: list, algorithm: str, normalize: bool, mode: str, plot: bool):

    deviation_rate = np.random.uniform(low=0.95, high=1.05, size=None) if normalize else None  # 5% of maximum deviation between classes
    if mode == 'train':
        logging.debug('Training')

        if models_dir_paths is None:
            models_dir_paths = []

        min_iterations = max(len(models_dir_paths), 1)  # we want to iterate through all given models or once if no model was given
        for idx in range(min_iterations):

            if models_dir_paths:
                with lzma.open(Path(models_dir_paths[idx], 'classifier.model'), 'rb') as model_file:
                    models, encoders = pickle.load(model_file)

                with open(Path(models_dir_paths[idx], 'parameters.json'), 'r') as params_file:
                    parameters = json.load(params_file)
            else:
                models = None
                encoders = None
                parameters = None

            classifier = ASRSReportClassificationTrainer(
                models=models,
                encoders=encoders,
                algorithm=algorithm,
                parameters=parameters,
                normalized=normalize,
                deviation_rate=deviation_rate
            )
            classifier.train_report_classification(texts_paths, label, label_filter, mode=mode)
    else:
        logging.debug(f'Testing on { "normalized " if normalize else "" }{ mode }')

        if models_dir_paths is None:
            raise ValueError("The path to the model cannot be null for testing")

        models_predictions = []
        test_targets = None
        for model_dir_path in models_dir_paths:

            with lzma.open(Path(model_dir_path, 'classifier.model'), 'rb') as model_file:
                models, encoders = pickle.load(model_file)
                logging.debug(f"Loaded {len(models)} models")

            with open(Path(model_dir_path, 'parameters.json'), 'r') as params_file:
                parameters = json.load(params_file)

            predictor = ASRSReportClassificationTrainer(
                models=models,
                encoders=encoders,
                parameters=parameters,
                normalized=normalize,
                deviation_rate=deviation_rate
            )

            if not texts_paths:
                texts_paths = [f'../ASRS/ASRS_{ mode }.csv']

            predictions, targets = predictor.train_report_classification(texts_paths, label, label_filter, mode=mode)
            models_predictions.append(predictions)
            if test_targets is None:
                test_targets = targets
        if plot:
            ASRSReportClassificationEvaluator.plot(models_predictions, test_targets)

        ASRSReportClassificationEvaluator.evaluate(models_predictions, test_targets)
