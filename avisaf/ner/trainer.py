#!/usr/bin/env python3
"""New entity trainer is the module responsible for creating a new or updating
and improving an existing Named Entity Recognition model. The module uses
train_spaCy_model function which updates the model using spaCy.
"""
import spacy
import logging
from typing import List
from datetime import datetime
from pathlib import Path
from spacy.tokens import DocBin
from spacy.cli.train import train
# importing own modules
from avisaf.util.data_extractor import get_entities
from .data_stream_registry import stream_data

logger = logging.getLogger("avisaf_logger")


class ASRSNamedEntityRecognizer:

    def __init__(self):
        self._pipe_name = "ner"
        self._default_nlp = spacy.load("en_core_web_md")  # loading default spacy language model

    def load_spacy_model(self, model: str = "en_core_web_md") -> spacy.Language:

        if model is None:
            logger.info("Using default spaCy NLP model.")
            return self._default_nlp

        try:
            nlp = spacy.load(model)
            logger.info(f"An already existing spaCy model has been loaded successfully: {model}.")
        except OSError:
            # using a blank English language spaCy model
            nlp = spacy.blank("en")
            logger.info("A new blank model has been created.")

        return nlp

    def setup_ner_pipeline(self, model):
        nlp = self.load_spacy_model(model)
        if not nlp.has_pipe(self._pipe_name):
            ner = nlp.add_pipe(self._pipe_name, last=True)
        else:
            ner = nlp.get_pipe(self._pipe_name)

        # getting a list of currently used entities from **default** location
        for label in list(get_entities().keys()):
            ner.add_label(label)

        return nlp, ner

    def convert_to_spacy_docbin(self, json_data: list, output_path: str = None) -> DocBin:
        docbin = DocBin()  # creating binary serialization representation of Docs collection

        logger.info("Starting JSON to DocBin conversion")
        for idx, (text, annotations) in enumerate(json_data):
            if idx % 1000 == 0:
                logger.debug(f"Converting {idx}")

            doc = self._default_nlp.make_doc(text)
            ents = []  # document's entities list
            for ent_start, ent_end, ent_label in annotations.get("entities"):
                named_entity = doc.char_span(ent_start, ent_end, label=ent_label)
                if not named_entity:
                    continue

                ents.append(named_entity)
            doc.ents = ents

            docbin.add(doc)

        if output_path is not None:
            logger.info(f"Saving converted DocBin collection to: {output_path}")
            docbin.to_disk(output_path)

        logger.info("Conversion completed")
        return docbin


def train_ner(
    config_file_path: str,
    model=None,
    new_model_name: str = None,
    train_data_srcfiles: List[Path] = None
):
    """SpaCy NER model training function. The function iterates given number of
    times over the given data in order to create an appropriate statistical
    entity prediction model.

    :param config_file_path:
    :type config_file_path: Union[str, Path]
    :type train_data_srcfiles: List[Union[Path, str]]
    :param train_data_srcfiles: A path to the files containing training data based
        based on which the spaCy model will be updated.
    :type model: Union[str, Path]
    :param model: The string representation of a spaCy model. Either existing
        pre-downloaded spaCy model or a path to a local directory.
    :type new_model_name: str
    :param new_model_name: New spaCy NER model will be saved under this name.
        This parameter also makes part of the path where the model will be
        saved.

    :return: Returns created NLP spaCy model.
    """

    ent_recognizer = ASRSNamedEntityRecognizer()

    nlp, ner = ent_recognizer.setup_ner_pipeline(model)

    if not train_data_srcfiles:
        logger.error("Missing training data path argument")
        return

    models_basedir = Path("models", "ner")
    models_basedir.mkdir(parents=True, exist_ok=True)
    if new_model_name is None:
        model_path = Path(models_basedir,  f"model_{datetime.today().strftime('%Y%m%d%H%M%S')}")
    else:
        model_path = Path(new_model_name).resolve()

    # extractor = JsonDataExtractor(train_data_srcfiles)
    # training_data = extractor.get_ner_training_data()

    # converted = ent_recognizer.convert_to_spacy_docbin(training_data, "test_data_11.spacy")

    overrides = {"corpora.train.data_source": train_data_srcfiles}

    if model is not None:
        config_file_path = str(Path("config", "spacy_ner_continue.cfg"))
        overrides.update({"components.ner.source": model})

    train(
        Path(config_file_path),
        model_path,
        overrides=overrides
    )
