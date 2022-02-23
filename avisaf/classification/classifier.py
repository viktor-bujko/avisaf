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
from pathlib import Path
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC

from training.training_data_creator import ASRSReportDataPreprocessor
from evaluation.tc_evaluator import ASRSReportClassificationEvaluator

logger = logging.getLogger(str(__file__))

logging.basicConfig(
    level=logging.DEBUG, format=f"[%(levelname)s - %(asctime)s]: %(message)s"
)


class ASRSReportClassificationPredictor:
    def __init__(
        self,
        models,
        vectorizer=None,
        normalized: bool = True,
        deviation_rate: float = 0.0,
        parameters=None,
    ):

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
            self._encodings = parameters[
                "encodings"
            ]  # "int: label" dictionary of possible classes
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
            raise ValueError(
                "A model needs to be trained or loaded first to perform predictions."
            )

        logger.info(f"Test data shape: {test_data.shape}")
        predictions = model.predict(test_data, None)

        return predictions

    @staticmethod
    def predict_proba(test_data, model=None):
        if model is None:
            raise ValueError(
                "A model needs to be trained or loaded first to perform predictions."
            )

        logger.info(f"Probability predictions made using model: {model}")
        if getattr(model, "predict_proba", None) is not None:
            predictions = model.predict_proba(test_data)
        else:
            predictions = model.predict(test_data)

        return predictions

    def _decode_prediction(self, prediction: int):
        if not len(self._encodings):
            raise ValueError("Train a model to get an non-empty encoding.")

        decoded_label = self._encodings.get(prediction)

        if decoded_label is None:
            raise ValueError(f'Encoding with value "{prediction}" does not exist.')

        return decoded_label

    def decode_predictions(self, predictions: list):
        if predictions is None:
            raise TypeError("Predictions have to be made first")

        vectorized = np.vectorize(self._decode_prediction)
        decoded_labels = vectorized(predictions)

        return decoded_labels

    def label_text(self, text):
        """

        :param text:
        :return:
        """
        if self._vectorizer is None:
            raise ValueError(
                "A model needs to be trained or loaded first to be able to transform texts."
            )

        vectorized_text = self._vectorizer.transform(text)
        prediction = self._models.predict(vectorized_text)
        predicted_label = self._decode_prediction(prediction)

        # TODO: For a given text, the classifier returns a dictionary containing field name as key and its predicted value
        return predicted_label


class ASRSReportClassificationTrainer:

    def _set_classification_algorithm(self, classification_algorithm: str):
        available_classifiers = {
            "mlp": MLPClassifier(
                hidden_layer_sizes=(512, 256),
                alpha=0.007,
                batch_size=128,
                learning_rate="adaptive",
                learning_rate_init=0.003,
                verbose=True,
                early_stopping=True,
                n_iter_no_change=20,
            ),
            "svm": LinearSVC(dual=False, class_weight="balanced"),
            "forest": RandomForestClassifier(
                n_estimators=150,
                criterion="entropy",
                min_samples_split=32,
                n_jobs=2,
                verbose=5,
            ),
            "knn": KNeighborsClassifier(n_neighbors=20, weights="distance"),
            "gauss": GaussianNB(),
            "mnb": MultinomialNB(),
            "regression": LogisticRegression(),
        }

        # Setting a default classifier value
        _classifier = available_classifiers["knn"]

        if available_classifiers.get(classification_algorithm) is not None:
            _classifier = available_classifiers[classification_algorithm]

        return _classifier

    def _get_encodings(self, parameters: dict):
        encodings = {}
        for label, encoding in parameters.get("encodings", {}).items():
            encodings.update(
                {label: {int(key): label_value for key, label_value in encoding.items()}}
            )
        return encodings

    def _restore_classifier_state(self, parameters: dict):
        for param, value in parameters.get("model_params", {}).items():
            try:
                setattr(self._classifier, param, value)
            except AttributeError:
                logging.warning(f"Trying to set a non-existing attribute {param} with value {value}")

    def __init__(
        self,
        models: dict = None,
        encoders: list = None,
        parameters: dict = None,
        algorithm=None,
        normalization: str = None,
        deviation_rate: float = 0.0,
    ):
        """

        :param models: Dictionary of (label, prediction_model) pairs. Label is a string representing the
                       topic to be predicted by the "prediction_model" - serialized scikit object.
        :param encoders: List of sklearn LabelEncoder objects for each model.
                         See https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
                         for more details.
        :param parameters: Dictionary of model parameters for better accessibility from ASRSTrainer.
        :param algorithm:  Text classification algorithm to be used.
        :param normalization: Training samples normalization method.
        :param deviation_rate:
        """
        if not parameters:
            parameters = {}

        self._normalize_method = normalization
        self._preprocessor = ASRSReportDataPreprocessor(encoders=encoders)

        if not models:
            self._classifier = self._set_classification_algorithm(algorithm)
            self._models = {}
            self._deviation_rate = 0.0
            self._encodings = {}
            self._model_params = self._classifier.get_params()
            self._params = {"algorithm": algorithm}
            self._trained_filtered_labels = {}
            self._trained_texts = []
        else:
            try:
                self._classifier = list(models.values())[0]  # extracting first scikit prediction object
                self._models = models
                self._deviation_rate = deviation_rate
                self._params = parameters
                self._encodings = self._get_encodings(parameters)
                # self._algorithm = parameters["algorithm"]
                self._model_params = parameters.get("model_params", {})
                self._trained_filtered_labels = parameters.get("trained_labels", {})
                self._trained_texts = parameters.get("trained_texts", [])
            except AttributeError:
                raise ValueError("Corrupted parameters.json file")

        assert self._models.keys() == self._encodings.keys()

        if self._classifier is not None and parameters.get("model_params") is not None:
            self._restore_classifier_state(parameters)

    def train_report_classification(
        self,
        texts_paths: list,
        label_to_train: str,
        label_filter: list = None,
        # mode: str = "dev",
    ):

        for text_path in texts_paths:
            # append information about trained text if not already present
            if text_path not in self._trained_texts:
                self._trained_texts.append(text_path)

        labels_to_train = list(self._trained_filtered_labels.keys())
        labels_values = list(self._trained_filtered_labels.values())

        if label_to_train is not None:
            labels_to_train = [label_to_train]

            if label_filter is None:
                # trying to get previously saved label filter
                if self._trained_filtered_labels.get(label_to_train):
                    labels_values = self._trained_filtered_labels[label_to_train]
                    filter_update = labels_values
                else:
                    labels_values = None
                    filter_update = []
            else:
                labels_values = [label_filter]
                filter_update = label_filter

            # if mode == "train":
            self._trained_filtered_labels.update({label_to_train: filter_update})

        if not labels_to_train:
            raise ValueError(
                "Nothing to train - please make sure at least one category is specified."
            )

        assert len(labels_to_train) == len(labels_values)

        labels_predictions, labels_targets = [], []

        data, target = self._preprocessor.vectorize_texts(
            texts_paths,
            labels_to_train,
            train=mode == "train",
            label_values_filter=labels_values,
            normalize=self._normalize_method,
        )

        for i, zipped in enumerate(zip(labels_to_train, labels_values)):

            lbl, fltr = zipped

            logger.debug(f"training data shape: {data[i].shape}")
            logger.debug(self._preprocessor.get_data_distribution(target[i])[1])

            if self._models.get(lbl) is not None:
                logging.debug("Found previously trained model")
                classifier = self._models[lbl]
                # TODO: setattr should be in try block
                setattr(classifier, "warm_start", True)
                setattr(classifier, "learning_rate_init", 0.0005)
            else:
                classifier = clone(self._classifier)

            # if mode == "train":
            # encoding is available only after texts vectorization
            encoding = {}
            for label_idx, label in enumerate(self._preprocessor.encoder(i).classes_):
                encoding.update({label_idx: label})
            self._encodings.update({lbl: encoding})

            self._params = {
                "algorithm": self._params.get("algorithm"),
                "encodings": self._encodings,
                "model_params": self._model_params,
                "trained_labels": self._trained_filtered_labels,
                "trained_texts": self._trained_texts,
                "vectorizer_params": self._preprocessor.vectorizer.get_params(),
            }

            classifier.fit(data[i], target[i])
            logging.info(f"MODEL: {classifier}")
            self._models.update({lbl: classifier})

            predictions = ASRSReportClassificationPredictor.predict_proba(data[i], classifier)
            labels_predictions.append(predictions)
            labels_targets.append(target[i])

            # if mode == "train":
            ASRSReportClassificationEvaluator.evaluate([[predictions]], [target[i]])

        # if mode == "train":
        self.save_model(self._models)

        return labels_predictions, labels_targets

    def save_model(self, models_to_save: dict):
        model_dir_name = "asrs_classifier-{}-{}-{}".format(
            self._params["algorithm"],
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            ",".join(
                (
                    "{}_{}".format(sub("(.)[^_]*_?", r"\1", key), value)
                    for key, value in sorted(self._model_params.items())
                )
            ).replace(" ", "_", -1),
        )

        model_dir_name = model_dir_name[:100]

        if self._normalize_method:
            model_dir_name += ",norm"

        Path("classifiers").mkdir(exist_ok=True)
        Path("classifiers", model_dir_name).mkdir(exist_ok=False)
        with lzma.open(
            Path("classifiers", model_dir_name, "classifier.model"), "wb"
        ) as model_file:
            logger.info(f"Saving {len(models_to_save)} model(s): {models_to_save}")
            pickle.dump((models_to_save, self._preprocessor.encoders), model_file)

        with open(
            Path("classifiers", model_dir_name, "parameters.json"),
            "w",
            encoding="utf-8",
        ) as params_file:
            logger.info(
                f"Saving parameters [encoding, model parameters, train_texts_paths, trained_labels, label_filter]"
            )
            json.dump(self._params, params_file, indent=4)


def train_classification(
    models_paths: list,
    texts_paths: list,
    label: str,
    label_values: list,
    algorithm: str,
    normalization: str
):
    """
    Method which sequentially launches the training of multiple text classification models.

    :param models_paths:  List of paths to the models which should be updated. If no such path is
                          provided, a new classification model will be created.
    :param texts_paths:   List of paths to the .csv files containing training examples.
    :param label:         Name of the topic being classified by the model.
    :param label_values:  List of possible classification classes for given label.
                          Actual values are determined by ASRS taxonomy.
    :param algorithm:     Classification algorithm used for classification.
    :param normalization: Training examples distribution normalization method. If None,
                          no data normalization is performed, therefore, target classes of
                          training examples may have uneven distribution. Currently supported
                          normalization methods are undersampling and oversampling of training
                          examples.
    """
    normalization_rate = (
        np.random.uniform(low=0.95, high=1.05, size=None) if normalization else None
    )  # 5% of maximum deviation between classes

    if not models_paths:
        classifier = ASRSReportClassificationTrainer(
            models=None,
            encoders=None,
            parameters=None,
            algorithm=algorithm,
            normalization=normalization,
            deviation_rate=normalization_rate
        )

        classifier.train_report_classification(texts_paths, label, label_values)
        return

    for model_path in models_paths:
        # Try to load the model and its parameters from idx-th model path
        with lzma.open(Path(model_path, "classifier.model"), "rb") as model_file,\
             open(Path(model_path, "parameters.json"), "r") as params_file:
            models, encoders = pickle.load(model_file)
            parameters = json.load(params_file)

        classifier = ASRSReportClassificationTrainer(
            models=models,
            encoders=encoders,
            parameters=parameters,
            algorithm=algorithm,
            normalization=normalization,
            deviation_rate=normalization_rate
        )
        classifier.train_report_classification(texts_paths, label, label_values)


def test_classification():
    pass


def launch_classification(
    models_dir_paths: list,
    texts_paths: list,
    label: str,
    label_filter: list,
    algorithm: str,
    normalize: str,
    mode: str,
    plot: bool,
):

    if mode == "train":
        logging.debug("Training")

        if models_dir_paths is None:
            models_dir_paths = []

        min_iterations = max(
            len(models_dir_paths), 1
        )  # we want to iterate through all given models or once if no model was given
        for idx in range(min_iterations):

            if models_dir_paths:
                with lzma.open(
                    Path(models_dir_paths[idx], "classifier.model"), "rb"
                ) as model_file:
                    models, encoders = pickle.load(model_file)

                with open(
                    Path(models_dir_paths[idx], "parameters.json"), "r"
                ) as params_file:
                    parameters = json.load(params_file)
            else:
                models = None
                encoders = None
                parameters = None

            classifier = ASRSReportClassificationTrainer(models=models, encoders=encoders, parameters=parameters,
                                                         algorithm=algorithm, normalization=normalize,
                                                         deviation_rate=deviation_rate)
            classifier.train_report_classification(
                texts_paths, label, label_filter, mode=mode
            )
    else:
        logging.debug(f'Testing on { "normalized " if normalize else "" }{ mode }')

        if models_dir_paths is None:
            raise ValueError("The path to the model cannot be null for testing")

        models_predictions = []
        test_targets = None
        for model_dir_path in models_dir_paths:

            with lzma.open(Path(model_dir_path, "classifier.model"), "rb") as model_file, open(Path(model_dir_path, "parameters.json"), "r") as params_file:
                models, encoders = pickle.load(model_file)
                logging.debug(f"Loaded {len(models)} models")
                parameters = json.load(params_file)

            # with open(Path(model_dir_path, "parameters.json"), "r") as params_file:
            #     parameters = json.load(params_file)

            predictor = ASRSReportClassificationTrainer(models=models, encoders=encoders, parameters=parameters,
                                                        normalization=normalize, deviation_rate=deviation_rate)

            if not texts_paths:
                texts_paths = [f"../ASRS/ASRS_{ mode }.csv"]

            predictions, targets = predictor.train_report_classification(
                texts_paths, label, label_filter, mode=mode
            )
            models_predictions.append(predictions)
            if test_targets is None:
                test_targets = targets
        if plot:
            ASRSReportClassificationEvaluator.plot(models_predictions, test_targets)

        ASRSReportClassificationEvaluator.evaluate(models_predictions, test_targets)
