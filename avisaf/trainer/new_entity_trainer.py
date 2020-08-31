#!/usr/bin/env python3

import spacy
import random
import json


def get_training_data(path):
    """

    :return:
    """
    with open(path, mode='r') as file:
        TR_DATA = json.loads(file.read())
        return TR_DATA


def trainer():
    nlp = spacy.blank('en')
    # nlp = spacy.load(os.path.expanduser('~/Documents/avisaf-ner/example-model'))

    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)

    for label in ['AIRPLANE', 'CREW', 'AIRPORT_TERM', 'FLIGHT_PHASE', 'AVIATION_TERM', 'NAV_WAYPOINT', 'ALTITUDE',
                  'WEATHER', 'ABBREVIATION']:
        ner.add_label(label)

    TRAINING_DATA = get_training_data('/home/viktor/Documents/avisaf_ner/generated_annotation_data.json')
    # Start the training
    optimizer = nlp.begin_training()

    # Iterate 15 times
    for itn in range(15):
        # Shuffle the data
        random.shuffle(TRAINING_DATA)
        losses = {}

        # Batch the examples and iterate over them
        for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
            texts = [text for text, entities in batch]
            entity_offsets = [entities for text, entities in batch]

            # Update the model
            nlp.update(texts, entity_offsets, sgd=optimizer, losses=losses)
        print(losses)

    nlp.to_disk('/home/viktor/Documents/avisaf-ner/newest-model')
    print('Model saved')


if __name__ == '__main__':
    trainer()
