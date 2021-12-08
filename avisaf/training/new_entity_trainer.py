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
from spacy.tokens import Doc
from spacy.pipeline import EntityRecognizer

# importing own modules
from typing import List

from avisaf.util.data_extractor import get_entities, get_training_data


@spacy.Language.component("aviation_ner")
def aviation_ner_component(doc: Doc) -> Doc:
    pass


def load_spacy_model(model: str = "en_core_web_md") -> spacy.Language:

    if model is None:
        return spacy.load("en_core_web_md")

    try:
        nlp = spacy.load(model)
        logging.info(f"An already existing spaCy model has been loaded successfully: {model}.")
    except OSError:
        # using a blank English language spaCy model
        nlp = spacy.blank("en")
        logging.info("A new blank model has been created.")

    return nlp


def train_spacy_ner(
    iter_number: int = 20,
    model=None,
    new_model_name: str = None,
    train_data_srcfiles: List[Path] = None,
    verbose: bool = False,
    batch_size: int = 256
):
    """SpaCy NER model training function. The function iterates given number of
    times over the given data in order to create an appropriate statistical
    entity prediction model.

    :type verbose: bool
    :param verbose: A flag indicating verbose stdout printing.
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
        # nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe(ner_pipe_name)

    # getting a list of currently used entities from **default** location
    for label in list(get_entities().keys()):
        ner.add_label(label)

    # Start the training
    optimizer = nlp.initialize() if model is None else nlp.resume_training()

    if verbose:
        print(f'Start time: {datetime.now().strftime("%H:%M:%S")}')
    start_time = time.time()

    if not train_data_srcfiles:
        print("Missing training data path argument", file=sys.stderr)
        return

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != ner_pipe_name]
    if new_model_name is None:
        model_path = Path("models", f"model_{datetime.today().strftime('%Y%m%d%H%M%S')}")
    else:
        model_path = Path("models", new_model_name).resolve()


    # Iterate iter_number times
    for itn in range(iter_number):
        print(f"Iteration: {itn}.")
        print(f"Model will be saved to the {model_path}_itn_{itn}")

        for train_data_file in train_data_srcfiles:
            train_data_file = Path(train_data_file)
            train_data_file = train_data_file if train_data_file.is_absolute() else train_data_file.resolve()

            logging.info(f"Using training dataset: {train_data_file}")
            training_data = get_training_data(train_data_file)

            random.shuffle(training_data)
            losses = {}
            start = time.time()

            with nlp.disable_pipes(*other_pipes):
                for batch in spacy.util.minibatch(training_data, size=batch_size):
                    # Get all the texts from the batch
                    # Get all entity annotations from the batch
                    examples = []
                    for text, ents in batch:
                        doc = nlp.make_doc(text)
                        example = Example.from_dict(doc, ents)
                        examples.append(example)

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
                            print(datetime.now().strftime("%H:%M:%S"), flush=verbose)
                            start = new_time

                    except ValueError as e:
                        print(e)
                        print(f"Exception occurred at: {datetime.now().strftime('%H:%M:%S')}",
                              f"for file: {train_data_srcfiles}.", file=sys.stderr)
                        sys.exit(1)

            nlp.to_disk(f"{model_path}_itn_{itn}")
            print(f"    Model saved successfully to {model_path}_itn_{itn}")
            print(f"    Losses for current train data file in iteration {itn}: {losses}.", flush=verbose)

    if verbose:
        print("Model saved")
        print(f"Execution time: {time.time() - start_time}")
        print(f'Finished at: {datetime.now().strftime("%H:%M:%S")}')

    return nlp
