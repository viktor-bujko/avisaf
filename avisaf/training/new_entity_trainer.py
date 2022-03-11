#!/usr/bin/env python3
"""New entity trainer is the module responsible for creating a new or updating
and improving an existing Named Entity Recognition model. The module uses
train_spaCy_model function which updates the model using spaCy.
"""
import logging
import sys
import spacy
import random
from datetime import datetime
import time
from pathlib import Path
from spacy.training import Example

# importing own modules
from typing import List
from util.data_extractor import get_entities, JsonDataExtractor

logger = logging.getLogger("avisaf_logger")


def load_spacy_model(model: str = "en_core_web_md") -> spacy.Language:

    if model is None:
        return spacy.load("en_core_web_md")

    try:
        nlp = spacy.load(model)
        logger.info(f"An already existing spaCy model has been loaded successfully: {model}.")
    except OSError:
        # using a blank English language spaCy model
        nlp = spacy.blank("en")
        logger.info("A new blank model has been created.")

    return nlp


def train_spacy_ner(
    iter_number: int = 20,
    model=None,
    new_model_name: str = None,
    train_data_srcfiles: List[Path] = None,
    batch_size: int = 256
):
    """SpaCy NER model training function. The function iterates given number of
    times over the given data in order to create an appropriate statistical
    entity prediction model.

    :type train_data_srcfiles: List[Union[Path, str]]
    :param train_data_srcfiles: A path to the files containing training data based
        based on which the spaCy model will be updated.
    :type iter_number: int
    :param iter_number: Number of iterations for NER model updating.
    :type model: str, Path
    :param model: The string representation of a spaCy model. Either existing
        pre-downloaded spaCy model or a path to a local directory.
    :type new_model_name: str
    :param new_model_name: New spaCy NER model will be saved under this name.
        This parameter also makes part of the path where the model will be
        saved.
    :type batch_size: int
    :param batch_size: Batch size to be used.

    :return: Returns created NLP spaCy model.
    """

    nlp = load_spacy_model(model)
    ner_pipe_name = "ner"
    if not nlp.has_pipe(ner_pipe_name):
        ner = nlp.add_pipe(ner_pipe_name, last=True)
    else:
        ner = nlp.get_pipe(ner_pipe_name)

    # getting a list of currently used entities from **default** location
    for label in list(get_entities().keys()):
        ner.add_label(label)

    # Start the training
    optimizer = nlp.initialize() if model is None else nlp.resume_training()

    logger.debug(f'Starting')
    start_time = time.time()

    if not train_data_srcfiles:
        logger.error("Missing training data path argument")
        return

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != ner_pipe_name]
    models_basedir = Path("models", "ner")
    if new_model_name is None:
        model_path = Path(models_basedir,  f"model_{datetime.today().strftime('%Y%m%d%H%M%S')}")
    else:
        model_path = Path(new_model_name).resolve()

    extractor = JsonDataExtractor(train_data_srcfiles)
    training_data = extractor.get_ner_training_data()

    # Iterate iter_number times
    for itn in range(iter_number):

        logger.info(f"Iteration: {itn}.")
        logger.info(f"Model will be saved to the {model_path}_itn_{itn}")

        random.shuffle(training_data)
        start = time.time()
        losses = {}

        with nlp.disable_pipes(*other_pipes):
            for batch in spacy.util.minibatch(training_data, size=batch_size):
                # Get all the texts from the batch
                # Get all entity annotations from the batch
                examples = []
                for text, ents in batch:
                    doc = nlp.make_doc(text)
                    doc_dict = {
                        "pos": [token.pos_ for token in doc],
                        "tags": [token.tag_ for token in doc],
                        "lemmas": [token.lemma_ for token in doc],
                        "deps": [token.dep_ for token in doc],
                        "text": doc.text
                    }
                    doc_dict.update(ents)
                    example = Example.from_dict(doc, doc_dict)
                    examples.append(example)
                logger.debug(f"Current examples count: {len(examples)}")
                try:
                    # Update the current model
                    nlp.update(
                        examples,
                        drop=0.3,
                        sgd=optimizer,
                        losses=losses
                    )
                    new_time = time.time()
                    if new_time - start > 60:
                        logger.info("Time information update")
                        start = new_time

                except ValueError as e:
                    logger.error(e)
                    logger.error(f"Exception occurred while processing file: {train_data_srcfiles}.")
                    sys.exit(1)

        nlp.to_disk(f"{model_path}_itn_{itn}")
        logger.info(f"\tModel saved successfully to {model_path}_itn_{itn}")
        logger.info(f"\tLosses for current train data file in iteration {itn}: {losses}.")

    logger.info("Model saved")
    logger.info(f"Execution time: {time.time() - start_time}")
    logger.info(f'Finished training.')

    return nlp
