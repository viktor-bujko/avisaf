#!/usr/bin/env python3

import sys
import os
import spacy
import random
from datetime import datetime
import time

PROJECT_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
#sys.path.append(PROJECT_ROOT_PATH)
sys.path.append('/home/viktor/Documents/avisaf_ner/avisaf')
sys.path.append('/home/viktor/Documents/avisaf_ner/avisaf/train')
sys.path.append('/home/viktor/Documents/avisaf_ner/avisaf/main')
sys.path.append('/home/viktor/Documents/avisaf_ner/avisaf/util')

from util.data_extractor import get_entities, get_training_data


def train_spaCy_model(iter_number=20,
                      model=None,
                      new_model_name=None,
                      train_data_srcfile=os.path.join(PROJECT_ROOT_PATH, 'data_files/auto_annotated_data.json'),
                      verbose=False):
    """

    :param verbose:
    :param train_data_srcfile:
    :param iter_number:
    :param model:
    :param new_model_name:
    :return:
    """
    if verbose:
        print('Start time.')
    start_time = time.time()
    if model is not None:
        nlp = spacy.load(model)
        print('An already existing spaCy model was successfully loaded.', flush=verbose)
    else:
        # using a blank English language spaCy model
        nlp = spacy.blank('en')
        print('A new blank model has been created.', flush=verbose)

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

    TRAINING_DATA = get_training_data(train_data_srcfile)

    with nlp.disable_pipes(*other_pipes):
        # Start the training
        optimizer = nlp.begin_training() if model is None else nlp.resume_training()

        # Iterate iter_number times
        for itn in range(iter_number):
            print(f'Iteration: {itn}.')
            # Shuffle the training data
            random.shuffle(TRAINING_DATA)
            losses = {}

            for batch in spacy.util.minibatch(TRAINING_DATA, size=3):
                # Get all the texts from the batch
                texts = [text for text, entities in batch]
                # Get all entity annotations from the batch
                entity_offsets = [entities for text, entities in batch]

                # Update the current model
                nlp.update(texts,
                           entity_offsets,
                           sgd=optimizer,
                           losses=losses)
                print(losses)
            print(f'Iteration {itn} losses: {losses}.', flush=verbose)

    if new_model_name is None:
        new_model_name = f"model_{datetime.today().strftime('%Y%m%d%H%M%S')}"

    model_path = os.path.join('/home/viktor/Documents/avisaf_ner/models', new_model_name)
    # os.path.join(PROJECT_ROOT_PATH, 'models', new_model_name)

    nlp.to_disk(model_path)
    if verbose:
        print('Model saved')
        print(f'Program execution time: {time.time() - start_time}')

    return nlp


if __name__ == '__main__':
    train_spaCy_model(model='/home/viktor/Documents/avisaf_ner/models/auto-generated-data-model',
                      new_model_name="auto-generated-data-model-1")
