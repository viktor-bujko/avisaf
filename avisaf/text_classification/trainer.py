#!/usr/bin/env python3
"""

"""

import lzma
import json
import logging
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC

from .predictor_decoder import ASRSReportClassificationPredictor, build_default_class_dict
from .data_preprocessor import ASRSReportDataPreprocessor
from .evaluator import ASRSReportClassificationEvaluator
from .visualizer import Visualizer
from avisaf.util.data_extractor import CsvAsrsDataExtractor

logger = logging.getLogger("avisaf_logger")


class ASRSReportClassificationTrainer:

    @staticmethod
    def _set_classification_algorithm(classification_algorithm: str):
        available_classifiers = {
            "mlp": MLPClassifier(
                hidden_layer_sizes=(512, 256),
                alpha=0.007,
                batch_size=128,
                learning_rate="adaptive",
                power_t=0.75,
                learning_rate_init=0.003,
                verbose=True,
                random_state=4030,
                early_stopping=True,
                n_iter_no_change=20,
            ),
            "svm": LinearSVC(dual=False, class_weight="balanced", verbose=5),
            "knn": KNeighborsClassifier(n_neighbors=20, weights="distance"),
        }

        # Setting a default classifier value
        classifier = available_classifiers.get(classification_algorithm, available_classifiers.get("mlp"))

        return classifier

    @staticmethod
    def _override_classifier_parameters(classifier, params_overrides: list):
        if not params_overrides:
            # nothing to override
            return

        for override in params_overrides:
            params = override.split("=", 1)
            assert len(params) == 2, f"Please make sure at least one \"=\" character is present. See --help for required format."
            param_key, param_value = params
            try:
                # casting string value to its type
                param_value = eval(param_value)
            except NameError:
                # if NameError is thrown - treat param_value as string (do nothing)
                pass
            logger.debug(f"Classifier parameter override: \"{param_key}\"={param_value}")
            setattr(classifier, param_key, param_value)

    @staticmethod
    def _get_encodings(parameters: dict):
        encodings = {}
        for label, encoding in parameters.get("encodings", {}).items():
            encodings.update(
                {label: {int(key): label_value for key, label_value in encoding.items()}}
            )
        return encodings

    @staticmethod
    def _restore_classifier_state(classifier, parameters: dict):
        for param, value in parameters.get("model_params", {}).items():
            try:
                setattr(classifier, param, value)
            except AttributeError:
                logger.warning(f"Trying to set a non-existing attribute {param} with value {value}")

    def _update_topic_model_encoding(self, lbl):
        """
        :param lbl: The label which has its encodings updated.
        """
        encoding = {}
        for label_idx, label in enumerate(self._preprocessor.encoder(lbl).classes_):
            encoding.update({label_idx: label})

        return encoding

    def _update_topic_params_dict(self, topic_label: str, params_dict: dict, new_params_dict: dict):
        # encoding is available only after texts vectorization
        for new_param_key, new_param_value in new_params_dict.items():
            # updating old parameters value or setting a new one if not exists
            params_dict.update({new_param_key: params_dict.get(new_param_key, new_param_value)})

        self._parameter_dicts.update({topic_label: params_dict})

    def __init__(
        self,
        models: [dict, None],
        encoders: [list, None],
        parameters: [dict, None],
        algorithm,
        vectorizer_type: str,
        normalization: str
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
        :param vectorizer_type:
        """
        if not parameters:
            parameters = {}

        self._algorithm = algorithm
        self._normalize_method = normalization
        self._classifier = self._set_classification_algorithm(algorithm)
        assert self._classifier is not None

        if not models:
            self._parameter_dicts = {}
            self._models = {}
            if not vectorizer_type:
                vectorizer_type = "tfidf"
        else:
            self._parameter_dicts = parameters
            self._models = models
            if not vectorizer_type:
                # taking first predictor's vectorizer name
                vectorizer_type = list(parameters.values())[0].get("vectorizer_params", {}).get("vectorizer", None)
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer=vectorizer_type, encoders=encoders)
        assert self._models.keys() == self._parameter_dicts.keys()

    def train_report_classification(self, texts_paths: list, label_to_train: str, label_filter: list = None, set_default: bool = False, params_overrides: list = None):
        """
        :param texts_paths:    The paths to the ASRS .csv files which are to be used as training examples sources.
        :param label_to_train: The text classification topic label to be trained.
        :param label_filter:   List of values to be used as classification classes for given topic label classification.
        :param set_default:    Boolean flag which specifies whether texts, which do not correspond to any of the value
                               defined in label_values list should still be included in training dataset with target
                               label "Other".
        :param params_overrides:
        """
        labels_to_train, labels_values = set_classifiers_to_train(label_to_train, label_filter, self._parameter_dicts)
        labels_predictions, labels_targets = [], []

        extractor = CsvAsrsDataExtractor(texts_paths)
        data, targets = self._preprocessor.extract_labeled_data(
            extractor,
            labels_to_extract=labels_to_train,
            label_classes_filter=labels_values,
            normalize=self._normalize_method,
            set_default=build_default_class_dict(labels_to_train, self._parameter_dicts, set_default),
        )

        model_dir_path = self._create_model_directory()
        for i, topic_label in enumerate(labels_to_train):
            # iterating through both lists
            train_data = (data[i]).astype(np.float)
            train_targets = targets[i].astype(np.int).ravel()

            logger.info(f"training data shape: {train_data.shape}")
            logger.info(self._preprocessor.get_data_targets_distribution(train_targets, label=topic_label)[1])

            if self._models.get(topic_label) is None:
                classifier = clone(self._classifier)
            else:
                # extracted classification model for given topic_label
                logger.debug("Found previously trained model")
                classifier = self._models.get(topic_label)
                setattr(classifier, "warm_start", True)

            self._restore_classifier_state(classifier, self._parameter_dicts.get(topic_label, {}))
            self._override_classifier_parameters(classifier, params_overrides)
            logger.info(f"MODEL: {classifier}")
            classifier = classifier.fit(train_data, train_targets)
            self._models.update({topic_label: classifier})

            topic_parameters = self._parameter_dicts.get(topic_label, {})

            trained_texts = topic_parameters.get("trained_texts", [])
            for text_path in texts_paths:
                # append information about trained text for given topic label if not already present
                if text_path not in trained_texts:
                    trained_texts.append(text_path)
            updated_encodings = self._update_topic_model_encoding(topic_label)

            dictionary_update = {
                "algorithm": self._algorithm,
                "has_default_class": set_default,
                "encodings": updated_encodings,
                "trained_labels": list(updated_encodings.values()),
                "vectorizer_params": self._preprocessor.vectorizer.get_params(),
                "model_params": classifier.get_params(),
                "trained_texts": trained_texts
            }
            self._update_topic_params_dict(topic_label, topic_parameters, dictionary_update)

            get_train_predictions = True
            if get_train_predictions:
                predictions = ASRSReportClassificationPredictor(extractor).get_model_predictions(classifier, train_data)
                label_encoder = self._preprocessor.encoder(topic_label)
                evaluator = ASRSReportClassificationEvaluator(None)
                evaluator.set_evaluated_topic_label(topic_label)
                evaluator.set_label_encoder(label_encoder)
                model_conf_matrix, model_results_dict = evaluator.evaluate(predictions, train_targets)
                visualizer = Visualizer(model_dir_path)
                visualizer.show_curves(predictions, train_targets, "train_data_model_prediction", topic_label=topic_label, label_encoder=self._preprocessor.encoder(topic_label))
                classes = label_encoder.inverse_transform(np.unique(train_targets))
                visualizer.print_metrics(f"Evaluating '{topic_label}' predictor on train data:", classes, model_conf_matrix, model_results_dict, "results_train")

        self.save_models(model_dir_path)

        return labels_predictions, labels_targets

    def _create_model_directory(self) -> str:
        model_dir_name = f"asrs_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self._normalize_method in self._preprocessor.normalization_methods:
            model_dir_name += self._normalize_method

        classifiers_dir = Path("models", "classifiers", self._algorithm if self._algorithm else ".")
        classifiers_dir.mkdir(exist_ok=True)
        model_dir_path = Path(classifiers_dir, model_dir_name)
        model_dir_path.mkdir(exist_ok=False)

        return str(model_dir_path)

    def save_models(self, model_dir_path: str):
        with lzma.open(Path(model_dir_path, "classifier.model"), "wb") as model_file:
            logger.info(f"Saving {len(self._models)} model(s): {self._models}")
            pickle.dump((self._models, self._preprocessor.label_encoders), model_file)

        with open(Path(model_dir_path, "parameters.json"), "w", encoding="utf-8") as params_file:
            logger.info(f"Saving model parameters")
            json.dump(self._parameter_dicts, params_file, indent=4)


def train_classification(models_paths: list, texts_paths: list, label: str, label_values: list, algorithm: str, normalization: str, set_default: bool, params_overrides: list, vectorizer_type: str):
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
    :param set_default:   Boolean flag which specifies whether texts, which do not correspond to
                          any of the value defined in label_values list should still be included
                          in training dataset with target label "Other".
    :param params_overrides:
    :param vectorizer_type:
    """

    if not models_paths:
        classifier = ASRSReportClassificationTrainer(
            models=None,
            encoders=None,
            parameters=None,
            algorithm=algorithm,
            vectorizer_type=vectorizer_type,
            normalization=normalization
        )

        _, _ = classifier.train_report_classification(texts_paths, label, label_values, set_default, params_overrides)
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
            vectorizer_type=vectorizer_type,
            normalization=normalization
        )
        _, _ = classifier.train_report_classification(texts_paths, label, label_values, set_default, params_overrides)


def set_classifiers_to_train(label_to_train: str = None, label_filter: list = None, parameter_dicts: dict = None):
    """
    Chooses the classifiers to be trained based on the given arguments. By default, all previously
    trained classifiers with saved classification classes are set to be trained again.

    :param label_to_train: Text classification topic label. This argument specifies the topic to
                           be trained by overriding the value to be returned.
    :param label_filter:   Values which represent possible classification classes for given
                           label_to_train.
    :param parameter_dicts:
    :return:               Tuple containing the list of topic labels based on which the
                           classifiers will be trained and the list containing corresponding
                           number of lists with classification classes for each item in labels_to_train list.
    """

    if not label_to_train:
        # setting default training values
        labels_to_train = list(parameter_dicts.keys())  # all text classification topic labels
        labels_values = []
        for topic_parameters in parameter_dicts.values():
            trained_labels = topic_parameters.get("trained_labels", [])
            labels_values.append(trained_labels)  # topic classification classes

        if not labels_to_train:
            raise ValueError("Nothing to train - please make sure at least one category is specified.")

        assert len(labels_to_train) == len(labels_values)
        return labels_to_train, labels_values

    # overriding label training settings
    assert label_to_train is not None
    if label_filter:
        return [label_to_train], [label_filter]

    # label_filter is not defined - trying to get previously saved label filter for given label_to_train
    labels_values = parameter_dicts.get(label_to_train, {}).get("trained_labels", [])

    return [label_to_train], labels_values
