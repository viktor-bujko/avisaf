#!/usr/bin/env python3

import sys
sys.path.append('/home/viktor/Documents/avisaf_ner/avisaf')

import spacy
import random
from util.data_extractor import get_entities, get_training_data
import os
from datetime import datetime


def train_spaCy_model(iter_number=20, model=None, new_model_name=None):
    """

    :param iter_number:
    :param model:
    :param new_model_name:
    :return:
    """

    if model is not None:
        nlp = spacy.load(model)
        print('An already existing spaCy model was successfully loaded.')
    else:
        # using a blank English language spaCy model
        nlp = spacy.blank('en')
        print('A new blank model has been created.')

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

    TRAINING_DATA = get_training_data('/home/viktor/Documents/avisaf_ner/resources/generated_annotation_data.json')

    with nlp.disable_pipes(*other_pipes):
        # Start the training
        optimizer = nlp.begin_training() if model is None else nlp.resume_training()

        # Iterate iter_number times
        for itn in range(iter_number):
            # Shuffle the training data
            random.shuffle(TRAINING_DATA)
            losses = {}

            # Divide examples into batches and iterate over them
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
            print(f'Current losses: {losses}.')

    if new_model_name is None:
        new_model_name = f"model_{datetime.today().strftime('%Y%m%d%H%M%S')}"

    model_path = os.path.expanduser('~/Documents/avisaf_ner/models/' + new_model_name)

    nlp.to_disk(model_path)
    print('Model saved')

    return nlp


if __name__ == '__main__':
    train_spaCy_model(model="/home/viktor/Documents/avisaf_ner/models/newest-model", new_model_name="retrained-model")

    """
    @plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
    )"""

    """
    matcher = PhraseMatcher(nlp.vocab)
        patterns = list(nlp.pipe(WORDS))
        matcher.add("ACFT_PARTS", None, *patterns)

        matches = matcher(doc)

        for (match_id, start, end) in matches:
            print(match_id, start, end, doc[start:end])

        for entity in doc.ents:
            print(entity.text, entity.label_)
    """