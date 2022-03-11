#!/usr/bin/env python3
"""

"""

import json
import lzma
import pickle
import logging
from pathlib import Path

import numpy as np

from training.training_data_creator import ASRSReportDataPreprocessor
from util.data_extractor import DataExtractor, CsvAsrsDataExtractor

logger = logging.getLogger("avisaf_logger")


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
    def __init__(self, data_extractor: DataExtractor, vectorizer: str = None):
        self._data_extractor = data_extractor
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer=vectorizer)

    def get_evaluation_predictions(self, prediction_models: dict, trained_labels: dict) -> list:

        # only asrs csv files are currently supported
        asrs_extractor = CsvAsrsDataExtractor(self._data_extractor.file_paths)
        data, targets = self._preprocessor.extract_labeled_data(
            asrs_extractor,
            labels_to_extract=list(trained_labels.keys()),
            label_classes_filter=list(trained_labels.values()),
            normalize=None  # do NOT change data distribution in prediction mode
        )

        all_predictions = []
        for topic_label, model, test_data, target in zip(trained_labels.keys(), prediction_models.values(), data, targets):
            logger.info(self._preprocessor.get_data_targets_distribution(target, label=topic_label)[1])
            predictions = self.get_model_predictions(model, test_data.astype(np.float))  # returns (samples, probabilities / one hot encoded predictions) shaped numpy array
            all_predictions.append((predictions, target.astype(np.int)))

        return all_predictions

    def get_all_predictions(self, prediction_models: dict) -> list:
        """
        :param prediction_models: Dictionary which contains (topic_label, model) items. topic_labels are the topics
                                  for which the model associated model predicts a class.
        :return: Numpy array of predictions for each prediction model i.e. array of matrices, where each matrix has
                (number_of_texts, number_of_classes_for_topic) shape.
        """

        all_predictions = []
        data = self._preprocessor.vectorize_texts(self._data_extractor)
        for predictor in prediction_models.values():
            predictions = self.get_model_predictions(predictor, data)
            all_predictions.append(predictions)

        return all_predictions

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
    predictor = ASRSReportClassificationPredictor(extractor, model_parameters.get("vectorizer_params", {}).get("vectorizer"))
    predictions = predictor.get_all_predictions(model_predictors)

    decoded_classes = ASRSReportClassificationDecoder(label_encoders).decode_predictions(predictions)
    # TODO: write structured form of text and predicted classes
    print(list(model_predictors.keys()))
    print(decoded_classes[:10])

    return predictions