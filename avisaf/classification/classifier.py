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
from util.data_extractor import DataExtractor
from evaluation.tc_evaluator import ASRSReportClassificationEvaluator
from evaluation.visualizer import Visualizer
from util.data_extractor import CsvAsrsDataExtractor

logger = logging.getLogger()

logging.basicConfig(format=f"[%(levelname)s - %(asctime)s]: %(message)s")


class ASRSReportClassificationDecoder:
    def __init__(self, encodings: list):
        self._encodings = encodings

    def decode_predictions(self, predictions: list):
        decoded_predictions = []

        for encoder, prediction in zip(self._encodings, predictions):
            predicted_classes = np.argmax(prediction, axis=1)
            original_encoding = encoder.inverse_transform(predicted_classes)
            decoded_predictions.append(np.reshape(original_encoding, (-1, 1)))

        decoded_predictions = np.concatenate(decoded_predictions, axis=1)
        return decoded_predictions


class ASRSReportClassificationPredictor:
    def __init__(self, data_extractor: DataExtractor):
        self._data_extractor = data_extractor
        self._preprocessor = ASRSReportDataPreprocessor()

    def get_evaluation_predictions(self, prediction_models: dict, trained_labels: dict) -> list:

        asrs_extractor = CsvAsrsDataExtractor(self._data_extractor.file_paths)  # only asrs csv files are supported
        data, targets = self._preprocessor.extract_labeled_data(
            asrs_extractor,
            labels_to_extract=list(trained_labels.keys()),
            label_values_filter=list(trained_labels.values())
        )

        all_predictions = []
        for model, test_data, target in zip(prediction_models.values(), data, targets):
            logging.info(self._preprocessor.get_data_distribution(target)[1])
            predictions = self.get_model_predictions(model, test_data)  # returns (samples, probabilities / one hot encoded predictions) shaped numpy array
            all_predictions.append((predictions, target))

        return all_predictions

    def get_all_predictions(self, prediction_models: dict) -> list:
        """
        :param prediction_models: Dictionary which contains (topic_label, model) items. topic_labels are the topics
                                  for which the model associated model predicts a class.
        :return: Numpy array of predictions for each prediction model i.e. array of matrices, where each matrix has
                (number_of_texts, number_of_classes_for_topic) shape.
        """

        narratives = self._data_extractor.extract_data(["Report 1_Narrative"]).get("Report 1_Narrative")
        data = self._preprocessor.vectorizer.build_feature_vectors(narratives)

        all_predictions = []
        for predictor in prediction_models.values():
            predictions = self.get_model_predictions(predictor, data)
            all_predictions.append(predictions)

        return all_predictions

    @staticmethod
    def get_model_predictions(prediction_model, data_vectors: np.array) -> np.array:
        if prediction_model is None:
            raise ValueError("A model needs to be trained or loaded first to perform predictions.")

        logging.info(f"Probability predictions made using model: {prediction_model}")
        if getattr(prediction_model, "predict_proba", None) is not None:
            predictions = prediction_model.predict_proba(data_vectors)
        else:
            predictions = prediction_model.predict(data_vectors)
            # TODO: convert predictions into one-hot representation

        return predictions


class ASRSReportClassificationTrainer:

    @staticmethod
    def _set_classification_algorithm(classification_algorithm: str):
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

    @staticmethod
    def _get_encodings(parameters: dict):
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

    def _set_classifiers_to_train(self, label_to_train: str = None, label_filter: list = None):
        """
        Chooses the classifiers to be trained based on the given arguments. By default, all previously
        trained classifiers with saved classification classes are set to be trained again.

        :param label_to_train: Text classification topic label. This argument specifies the topic to
                               be trained by overriding the value to be returned.
        :param label_filter:   Values which represent possible classification classes for given
                               label_to_train.
        :return:               Tuple containing the list of topic labels based on which the
                               classifiers will be trained and the list containing corresponding
                               number of lists with classification classes for each item in labels_to_train list.
        """
        # setting default training values
        labels_to_train = list(self._trained_filtered_labels.keys())  # all text classification topic labels
        labels_values = list(self._trained_filtered_labels.values())  # topic classification classes

        if label_to_train is not None:
            # overriding label training settings
            labels_to_train = [label_to_train]

            if label_filter is None:
                # trying to get previously saved label filter for given label_to_train
                if self._trained_filtered_labels.get(label_to_train):
                    labels_values = self._trained_filtered_labels.get(label_to_train)
                    filter_update = labels_values
                else:
                    labels_values = None
                    filter_update = []
            else:
                labels_values = [label_filter]
                filter_update = label_filter

            self._trained_filtered_labels.update({label_to_train: filter_update})

        if not labels_to_train:
            raise ValueError("Nothing to train - please make sure at least one category is specified.")

        assert len(labels_to_train) == len(labels_values)

        return labels_to_train, labels_values

    def _update_model_encoding(self, idx: int, lbl):
        """
        :param idx: Integer index of an encoder for idx-th classifier model.
        :param lbl: The label which has its encodings updated.
        """
        encoding = {}
        for label_idx, label in enumerate(self._preprocessor.encoder(idx).classes_):
            encoding.update({label_idx: label})

        self._encodings.update({lbl: encoding})

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
                self._model_params = parameters.get("model_params", {})
                self._trained_filtered_labels = parameters.get("trained_labels", {})
                self._trained_texts = parameters.get("trained_texts", [])
            except AttributeError:
                raise ValueError("Corrupted parameters.json file")

        assert self._models.keys() == self._encodings.keys()

        if self._classifier is not None and parameters.get("model_params") is not None:
            self._restore_classifier_state(parameters)

    def train_report_classification(self, texts_paths: list, label_to_train: str, label_filter: list = None):
        """
        :param texts_paths: The paths to the ASRS .csv files which are to be used as training examples sources.
        :param label_to_train: The text classification topic label to be trained.
        :param label_filter:   List of values to be used as classification classes for given topic label classification.
        """
        labels_to_train, labels_values = self._set_classifiers_to_train(label_to_train, label_filter)
        labels_predictions, labels_targets = [], []

        extractor = CsvAsrsDataExtractor(texts_paths)
        data, target = self._preprocessor.extract_labeled_data(
            extractor,
            labels_to_train,
            label_values_filter=labels_values,
            normalize=self._normalize_method,
        )

        for text_path in texts_paths:
            # append information about trained text if not already present
            if text_path not in self._trained_texts:
                self._trained_texts.append(text_path)

        for i, zipped in enumerate(zip(labels_to_train, labels_values)):
            # iterating through both lists

            lbl, fltr = zipped

            logging.debug(f"training data shape: {data[i].shape}")
            logging.debug(self._preprocessor.get_data_distribution(target[i])[1])

            if self._models.get(lbl) is None:
                classifier = clone(self._classifier)
            else:
                # extracted classification model for given topic label "lbl"
                logging.debug("Found previously trained model")
                classifier = self._models.get(lbl)
                setattr(classifier, "warm_start", True)
                setattr(classifier, "learning_rate_init", 0.0005)

            # encoding is available only after texts vectorization
            self._update_model_encoding(i, lbl)
            self._params = {
                "algorithm": self._params.get("algorithm"),
                "encodings": self._encodings,
                "model_params": self._model_params,
                "trained_labels": self._trained_filtered_labels,
                "trained_texts": self._trained_texts,
                "vectorizer_params": self._preprocessor.vectorizer.get_params(),
            }

            logging.info(f"MODEL: {classifier}")
            classifier.fit(data[i], target[i])
            self._models.update({lbl: classifier})

            get_train_predictions = True
            if get_train_predictions:
                predictions = ASRSReportClassificationPredictor(extractor).get_model_predictions(classifier, data[i])
                evaluator = ASRSReportClassificationEvaluator(lbl)
                model_conf_matrix, model_results_dict = evaluator.evaluate(predictions, target[i], self._encodings.get(lbl))
                Visualizer().print_metrics(f"Evaluating '{lbl}' predictor on training data:", model_conf_matrix, model_results_dict)

        self.save_models()

        return labels_predictions, labels_targets

    def save_models(self):
        model_dir_name = "asrs_classifier-{}-{}-{}".format(
            self._params.get("algorithm", ""),
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            ",".join(
                "{}_{}".format(sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(self._model_params.items())
            ).replace(" ", "_", -1),
        )[:100]

        if self._normalize_method == "oversample/undersample":
            # TODO: change to correct values
            model_dir_name += ",norm"

        classifiers_dir = Path("models", "classifiers")
        classifiers_dir.mkdir(exist_ok=True)
        model_dir_path = Path(classifiers_dir, model_dir_name)
        model_dir_path.mkdir(exist_ok=False)
        with lzma.open(Path(model_dir_path, "classifier.model"), "wb") as model_file:
            logging.info(f"Saving {len(self._models)} model(s): {self._models}")
            pickle.dump((self._models, self._preprocessor.encoders), model_file)

        with open(Path(model_dir_path, "parameters.json"), "w", encoding="utf-8") as params_file:
            logging.info(f"Saving parameters [encoding, model parameters, train_texts_paths, trained_labels, label_filter]")
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
    normalization_rate = (np.random.uniform(low=0.95, high=1.05, size=None) if normalization else None)  # 5% of maximum deviation between classes

    if not models_paths:
        classifier = ASRSReportClassificationTrainer(
            models=None,
            encoders=None,
            parameters=None,
            algorithm=algorithm,
            normalization=normalization,
            deviation_rate=normalization_rate
        )

        _, _ = classifier.train_report_classification(texts_paths, label, label_values)
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
        _, _ = classifier.train_report_classification(texts_paths, label, label_values)


def test_classification(model_path: str, text_paths: list, decode: bool = False, evaluate: bool = True, show_curves: bool = False):
    """

    :param decode:
    :param model_path:
    :param text_paths:
    :return:
    """
    if not model_path or not text_paths:
        logging.error("Both model_path and text_path arguments must be specified")
        return

    with lzma.open(Path(model_path, "classifier.model"), "rb") as model_file,\
         open(Path(model_path, "parameters.json"), "r") as params:
        model_predictors, label_encoders = pickle.load(model_file)
        params = json.load(params)

    extractor = CsvAsrsDataExtractor(text_paths)
    predictor = ASRSReportClassificationPredictor(extractor)

    if evaluate:
        predictions_targets = predictor.get_evaluation_predictions(model_predictors, params.get("trained_labels"))
        for (predictions, targets), topic_label, label_encoder in zip(predictions_targets, model_predictors.keys(), label_encoders):
            evaluator = ASRSReportClassificationEvaluator(topic_label, label_encoder)
            model_conf_matrix, model_results_dict = evaluator.evaluate(predictions, targets, show_curves=show_curves)
            Visualizer().print_metrics(f"Evaluating '{topic_label}' predictor:", model_conf_matrix, model_results_dict)
            evaluator.evaluate_dummy_baseline(targets, show_curves=show_curves)
            evaluator.evaluate_random_predictions(targets, show_curves=show_curves)
        return

    predictions = predictor.get_all_predictions(model_predictors)

    if decode:
        decoded_classes = ASRSReportClassificationDecoder(label_encoders).decode_predictions(predictions)
        print(list(model_predictors.keys()))
        print(decoded_classes[:10])

    return predictions


"""
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

        min_iterations = max(len(models_dir_paths), 1)  # we want to iterate through all given models or once if no model was given
        for idx in range(min_iterations):

            if models_dir_paths:
                with lzma.open(Path(models_dir_paths[idx], "classifier.model"), "rb") as model_file:
                    models, encoders = pickle.load(model_file)

                with open(Path(models_dir_paths[idx], "parameters.json"), "r") as params_file:
                    parameters = json.load(params_file)
            else:
                models = None
                encoders = None
                parameters = None

            classifier = ASRSReportClassificationTrainer(models=models, encoders=encoders, parameters=parameters,
                                                         algorithm=algorithm, normalization=normalize,
                                                         deviation_rate=0.0)
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

            predictor = ASRSReportClassificationTrainer(models=models, encoders=encoders, parameters=parameters, normalization=normalize, deviation_rate=0.0)

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
"""