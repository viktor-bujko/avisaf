#!/usr/bin/env python3
"""

"""

import json
import lzma
import pickle
import logging
import numpy as np
from pathlib import Path

from .data_preprocessor import ASRSReportDataPreprocessor
from .vectorizers import VectorizerFactory
import avisaf.util.data_extractor as de

logger = logging.getLogger("avisaf_logger")


def build_default_class_dict(topic_labels: list, parameter_dicts: dict, set_default: bool):
    default_class_dict = {}

    for topic_label in topic_labels:
        set_default_class = parameter_dicts.get(topic_label, {}).get("has_default_class", set_default)
        default_class_dict.update({topic_label: set_default_class})

    return default_class_dict


class ASRSReportClassificationDecoder:
    def __init__(self, encodings: dict):
        self._encodings = encodings

    def decode_predictions(self, predictions: list):
        if not predictions:
            return [{}]

        decoded_predictions = [{} for _ in range(len(predictions[0]))]

        for (encoder_name, encoder), prediction in zip(self._encodings.items(), predictions):
            predicted_classes = np.argmax(prediction, axis=1)
            original_encoding = encoder.inverse_transform(predicted_classes)
            for dict_idx, decoded_prediction in enumerate(original_encoding):
                decoded_predictions[dict_idx].update({encoder_name: decoded_prediction})
            # decoded_predictions.append(np.reshape(original_encoding, (-1, 1)))

        # decoded_predictions = np.concatenate(decoded_predictions, axis=1)
        return decoded_predictions


class ASRSReportClassificationPredictor:
    def __init__(self, data_extractor: de.DataExtractor):
        self._data_extractor = data_extractor
        # preprocessor which uses default vectorizer
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer="default")

    def get_evaluation_predictions(self, prediction_models: dict, models_parameters: dict) -> list:

        # only asrs csv files are currently supported __FOR EVALUATION__
        asrs_extractor = de.CsvAsrsDataExtractor(self._data_extractor.file_paths)
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
            if new_vectorizer is None:
                logger.info("Reusing default vectorizer")
            else:
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
            logger.info(self._preprocessor.get_data_targets_distribution(target, label=topic_label)[1])
            predictions = self.get_model_predictions(model, test_data.astype(np.float))  # returns (samples, probabilities / one hot encoded predictions) shaped numpy array
            all_predictions.append((predictions, target.astype(np.int)))

        return all_predictions

    def get_all_predictions(self, prediction_models: dict, models_parameters: dict) -> tuple:
        """
        :param prediction_models: Dictionary which contains (topic_label, model) items. topic_labels are the topics
                                  for which the model associated model predicts a class.
        :param models_parameters:
        :return: Numpy array of predictions for each prediction model i.e. array of matrices, where each matrix has
                (number_of_texts, number_of_classes_for_topic) shape.
        """
        all_predictions = []
        default_data = self._preprocessor.vectorizer.vectorize_texts(self._data_extractor)  # using default vectorizer
        narratives = self._preprocessor.vectorizer.vectorize_texts(self._data_extractor, return_vectors=False)

        for topic_label, predictor in prediction_models.items():
            new_vectorizer = self._data_need_revectoring(models_parameters.get(topic_label, {}))
            if new_vectorizer is None:
                # new_vectorizer is None -> data vectorized by default vectorizer can be used.
                data = default_data
            else:
                # current predictor uses different vectorization method than the default
                data = new_vectorizer.vectorize_texts(self._data_extractor)
            predictions = self.get_model_predictions(predictor, data)
            all_predictions.append(predictions)

        return narratives, all_predictions

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
            # we expect to predict each desired class at least once
            one_hot_predictions = np.zeros((predictions.shape[0], np.unique(predictions).shape[0]))
            for idx, pred in enumerate(predictions):
                # arbitrarily chosen confidence value of 100 % = 1
                one_hot_predictions[idx, pred] = 1
            predictions = one_hot_predictions

        return predictions


def launch_classification(model_path: str, text_path: Path, text: str):
    """
    :param model_path:
    :param text_path:
    :param text:
    :return:
    """
    if not model_path or (not text_path and not text):
        # TODO:
        logger.error("Both model_path and text(s) to be used must be specified")
        return

    with lzma.open(Path(model_path, "classifier.model"), "rb") as model_file, \
         open(Path(model_path, "parameters.json"), "r") as params_file:
        model_predictors, label_encoders = pickle.load(model_file)
        model_parameters = json.load(params_file)

    if text:
        extractor = de.PlainTextExtractor(text)
    else:
        text_path = Path(text_path)
        if text_path.suffix == ".csv":
            extractor = de.CsvAsrsDataExtractor([text_path])
        else:
            extractor = de.TextFileExtractor([text_path])

    predictor = ASRSReportClassificationPredictor(extractor)
    narratives, predictions = predictor.get_all_predictions(model_predictors, model_parameters)

    decoded_classes = ASRSReportClassificationDecoder(label_encoders).decode_predictions(predictions)
    for idx, (narrative, report_predictions_dict) in enumerate(zip(narratives, decoded_classes)):
        sep = "=" * 20
        msg = f"Report { idx + 1 } predictions"
        print(sep, msg, sep)
        print("Report narrative:", narrative, "", sep="\n")
        print("Predictions:")
        for (label, prediction_class) in report_predictions_dict.items():
            print(f"\t{label}: {prediction_class}")
        print(sep, "=" * len(msg), sep)

    return predictions
