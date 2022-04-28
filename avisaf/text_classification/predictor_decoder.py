#!/usr/bin/env python3
"""

"""

import json
import lzma
import pickle
import logging
import numpy as np
from pathlib import Path

from ner.annotator import ASRSReportDataPreprocessor
from util.vectorizers import VectorizerFactory
from util.data_extractor import DataExtractor, CsvAsrsDataExtractor

logger = logging.getLogger("avisaf_logger")


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


def build_default_class_dict(topic_labels: list, parameter_dicts: dict, set_default: bool):
    default_class_dict = {}

    for topic_label in topic_labels:
        set_default_class = parameter_dicts.get(topic_label, {}).get("has_default_class", set_default)
        default_class_dict.update({topic_label: set_default_class})

    return default_class_dict


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
        # preprocessor which uses default vectorizer
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer="default")

    def get_evaluation_predictions(self, prediction_models: dict, models_parameters: dict) -> list:

        # only asrs csv files are currently supported
        asrs_extractor = CsvAsrsDataExtractor(self._data_extractor.file_paths)
        vectorizer_name = list(models_parameters.values())[0].get("vectorizer_params", {}).get("vectorizer", None)
        self._preprocessor.vectorizer = VectorizerFactory.create_vectorizer(vectorizer_name)
        logger.debug(f"Using {vectorizer_name} vectorizer")
        data, targets = self._preprocessor.extract_labeled_data(
            asrs_extractor,
            labels_to_extract=list(models_parameters.keys()),
            label_classes_filter=[model_params.get("trained_labels", []) for model_params in models_parameters.values()],
            normalize=None,  # do NOT change data distribution in prediction mode
            set_default=build_default_class_dict(list(models_parameters.keys()), models_parameters, False)
        )

        all_predictions = []
        for (topic_label, model), test_data, target in zip(prediction_models.items(), data, targets):
            new_vectorizer = self._data_need_revectoring(models_parameters.get(topic_label, {}))
            if new_vectorizer is not None:
                logger.info("Creating new vectors for data")
                self._preprocessor.vectorizer = new_vectorizer
                d, t = self._preprocessor.extract_labeled_data(
                    asrs_extractor,
                    labels_to_extract=[topic_label],
                    label_classes_filter=[models_parameters.get(topic_label, {}).get("trained_labels", [])],
                    normalize=None,
                    set_default={topic_label: models_parameters.get(topic_label, {}).get("has_default_class", False)}
                )
                test_data = d[0]
                target = t[0]
            else:
                logger.info("Reusing default vectorizer")
            logger.info(self._preprocessor.get_data_targets_distribution(target, label=topic_label)[1])
            predictions = self.get_model_predictions(model, test_data.astype(np.float))  # returns (samples, probabilities / one hot encoded predictions) shaped numpy array
            all_predictions.append((predictions, target.astype(np.int)))

        return all_predictions

    def get_all_predictions(self, prediction_models: dict, models_parameters: dict) -> list:
        """
        :param prediction_models: Dictionary which contains (topic_label, model) items. topic_labels are the topics
                                  for which the model associated model predicts a class.
        :param models_parameters:
        :return: Numpy array of predictions for each prediction model i.e. array of matrices, where each matrix has
                (number_of_texts, number_of_classes_for_topic) shape.
        """
        all_predictions = []
        default_data = self._preprocessor.vectorizer.vectorize_texts(self._data_extractor)  # using default vectorizer

        for topic_label, predictor in prediction_models.items():
            new_vectorizer = self._data_need_revectoring(models_parameters.get(topic_label, {}))
            if new_vectorizer is not None:
                # current predictor uses different vectorization method than the default
                data = new_vectorizer.vectorize_texts(self._data_extractor)
            else:
                # new_vectorizer is None -> data vectorized by default vectorizer can be used.
                data = default_data
            predictions = self.get_model_predictions(predictor, data)
            all_predictions.append(predictions)

        return all_predictions

    def _data_need_revectoring(self, model_parameters: dict):
        default_vectorizer_params = self._preprocessor.vectorizer.get_params()
        predictor_vectorizer_params = model_parameters.get("vectorizer_params", None)
        if not predictor_vectorizer_params:
            logger.warning("Current predictor vectorizer parameters are unknown.")
            return None
        if predictor_vectorizer_params == default_vectorizer_params:
            # current vectorizer can be used
            return None

        # getting information which vectorizer current predictor has been trained with
        new_vectorizer = VectorizerFactory.create_vectorizer(predictor_vectorizer_params.get("vectorizer"))
        if new_vectorizer.get_params() == default_vectorizer_params:
            logger.warning("New vectorizer has same parameters as the old one. Reusing previous.")
            return None

        return new_vectorizer

    @staticmethod
    def get_model_predictions(prediction_model, data_vectors: np.array) -> np.array:
        if prediction_model is None:
            raise ValueError("A model needs to be trained or loaded first to perform predictions.")

        logger.info(f"Probability predictions made using model: {prediction_model}")
        if getattr(prediction_model, "predict_proba", None) is not None:
            predictions = prediction_model.predict_proba(data_vectors)
        else:
            predictions = prediction_model.predict(data_vectors)
            one_hot_predictions = np.zeros((predictions.shape[0], np.unique(predictions).shape[0]))  # we expect to predict each desired class at least once
            for idx, pred in enumerate(predictions):
                # arbitrarily chosen confidence value of 100 % = 1
                one_hot_predictions[idx, pred] = 1
            predictions = one_hot_predictions

        return predictions


def launch_classification(model_path: str, text_paths: list):
    """
    :param model_path:
    :param text_paths:
    :return:
    """
    if not model_path or not text_paths:
        logger.error("Both model_path and text_path arguments must be specified")
        return

    with lzma.open(Path(model_path, "classifier.model"), "rb") as model_file, \
         open(Path(model_path, "parameters.json"), "r") as params_file:
        model_predictors, label_encoders = pickle.load(model_file)
        model_parameters = json.load(params_file)

    extractor = CsvAsrsDataExtractor(text_paths)
    predictor = ASRSReportClassificationPredictor(extractor)
    predictions = predictor.get_all_predictions(model_predictors, model_parameters)

    decoded_classes = ASRSReportClassificationDecoder(label_encoders).decode_predictions(predictions)
    # TODO: write structured form of text and predicted classes
    print(list(model_predictors.keys()))
    print(decoded_classes[:10])

    return predictions
