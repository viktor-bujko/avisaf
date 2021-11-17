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
from spacy.training import Example, offsets_to_biluo_tags


# importing own modules
from avisaf.util.data_extractor import get_entities, get_training_data


def train_spacy_ner(
    iter_number: int = 20,
    model=None,
    new_model_name: str = None,
    tr_data_srcfile: Path = None,
    verbose: bool = False,
    batch_size: int = 256
):
    """SpaCy NER model training function. The function iterates given number of
    times over the given data in order to create an appropriate statistical
    entity prediction model.

    :type verbose: bool
    :param verbose: A flag indicating verbose stdout printing.
    :type tr_data_srcfile: Path
    :param tr_data_srcfile: A path to the file containing training data based
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

    if not tr_data_srcfile:
        print("Missing training data path argument", file=sys.stderr)
        return

    tr_data_srcfile = (
        Path(tr_data_srcfile)
        if tr_data_srcfile.is_absolute()
        else tr_data_srcfile.resolve()
    )
    if verbose:
        print(f'Start time: {datetime.now().strftime("%H:%M:%S")}')
    start_time = time.time()
    try:
        nlp = spacy.load(model)
        logging.info(f"An already existing spaCy model has been loaded successfully: {model}.")
    except OSError:
        # using a blank English language spaCy model
        nlp = spacy.blank("en")
        logging.info("A new blank model has been created.")

    logging.info(f"Using training dataset: {tr_data_srcfile}")

    # getting a list of currently used entities from **default** location
    entity_labels = list(get_entities().keys())

    if not nlp.has_pipe("ner"):
        ner = nlp.add_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    other_pipe_names = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    for label in entity_labels:
        ner.add_label(label)

    training_data = get_training_data(tr_data_srcfile)

    # Start the training
    optimizer = nlp.initialize() if model is None else nlp.resume_training()

    # Iterate iter_number times
    for itn in range(iter_number):
        print(f"Iteration: {itn}.")

        random.shuffle(training_data)
        losses = {}
        start = time.time()

        if new_model_name is None:
            new_model_name = f"model_{datetime.today().strftime('%Y%m%d%H%M%S')}"

        model_path = str(Path("models", new_model_name).resolve())

        with nlp.disable_pipes(*other_pipe_names):
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
                    print(f"Exception occurred at: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"for file: {tr_data_srcfile}.", file=sys.stderr)
                    sys.exit(1)

        nlp.to_disk(model_path)
        print(f"Model saved successfully to {model_path}")
        print(f"Iteration {itn} losses: {losses}.", flush=verbose)

    if verbose:
        print("Model saved")
        print(f"Execution time: {time.time() - start_time}")
        print(f'Finished at: {datetime.now().strftime("%H:%M:%S")}')

    return nlp
