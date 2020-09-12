#!/usr/bin/env python3

import sys
import spacy
import random
from datetime import datetime
import time
from pathlib import Path

SOURCES_ROOT_PATH = Path(__file__).parent.parent.resolve()
PROJECT_ROOT_PATH = SOURCES_ROOT_PATH.parent.resolve()
sys.path.append(str(SOURCES_ROOT_PATH))

from util.data_extractor import get_entities, get_training_data


def train_spaCy_model(iter_number: int = 20,
                      model=None,
                      new_model_name: str = None,
                      tr_data_srcfile: Path = Path(PROJECT_ROOT_PATH, 'data_files', 'training', 'auto_annotated_data.json').resolve(),
                      verbose: bool = False):
    """

    :type verbose:          bool
    :param verbose:
    :type tr_data_srcfile:  Path
    :param tr_data_srcfile:
    :type iter_number:      int
    :param iter_number:
    :type model:            str, Path
    :param model:
    :type new_model_name:   str
    :param new_model_name:
    :return:
    """

    given_data_src = str(tr_data_srcfile)
    tr_data_srcfile = Path(tr_data_srcfile) if tr_data_srcfile.is_absolute() else Path(tr_data_srcfile).resolve()
    if verbose:
        print(f'Start time: {datetime.now().strftime("%H:%M:%S")}')
    start_time = time.time()
    try:
        nlp = spacy.load(model)
        print(f'An already existing spaCy model was successfully loaded: {model}.', flush=verbose, file=sys.stderr)
    except OSError:
        # using a blank English language spaCy model
        nlp = spacy.blank('en')
        print('A new blank model has been created.', flush=verbose, file=sys.stderr)

    print(f'Using training dataset: {given_data_src}', flush=verbose)
    # getting a list of currently used entities from default location
    entity_labels = get_entities()

    if not nlp.has_pipe('ner'):
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    for label in entity_labels:
        ner.add_label(label)

    TRAINING_DATA = get_training_data(tr_data_srcfile)

    with nlp.disable_pipes(*other_pipes):
        # Start the training
        optimizer = nlp.begin_training() if model is None else nlp.resume_training()

        # Iterate iter_number times
        for itn in range(iter_number):
            print(f'Iteration: {itn}.')

            random.shuffle(TRAINING_DATA)
            losses = {}
            start = time.time()

            for batch in spacy.util.minibatch(TRAINING_DATA, size=3):
                # Get all the texts from the batch
                texts = [text for text, entities in batch]
                # Get all entity annotations from the batch
                entity_offsets = [entities for text, entities in batch]

                try:
                    # Update the current model
                    nlp.update(texts,
                               entity_offsets,
                               sgd=optimizer,
                               losses=losses)
                    new_time = time.time()
                    if new_time - start > 60:
                        print(datetime.now().strftime("%H:%M:%S"), flush=verbose)
                        start = new_time
                except ValueError as e:
                    print(e)
                    print("The optimizer probably couldn't create better prediction.", file=sys.stderr)
                    print(f"Exception occurred at: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"for file: {given_data_src}.", file=sys.stderr)
                    sys.exit(1)

            print(f'Iteration {itn} losses: {losses}.', flush=verbose)

    if new_model_name is None:
        new_model_name = f"model_{datetime.today().strftime('%Y%m%d%H%M%S')}"

    model_path = str(Path(PROJECT_ROOT_PATH, 'models', new_model_name).resolve())

    nlp.to_disk(model_path)
    if verbose:
        print('Model saved')
        print(f'Execution time: {time.time() - start_time}')
        print(f'Finished at: {datetime.now().strftime("%H:%M:%S")}')

    return nlp


if __name__ == '__main__':
    """model = None if sys.argv[1] == "None" else sys.argv[1]
    new_name = sys.argv[2]
    data_src = sys.argv[3]
    try:
        if sys.argv[4]:
            verbose = True
        else:
            verbose = False
    except IndexError:
        verbose = False"""

    """train_spaCy_model(model=model,  # '/home/viktor/Documents/avisaf_ner/models/auto-generated-data-model',
                      new_model_name=new_name,
                      train_data_srcfile=data_src,
                      verbose=verbose)"""
