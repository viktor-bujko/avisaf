#!/usr/bin/env python3

import spacy
from util.indexing import get_training_data
import os
import random


def trainer():

    # nlp = spacy.blank('en')
    nlp = spacy.load(os.path.expanduser('~/Documents/avisaf-ner/example-model'))

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)

        ner.add_label('ACFT_PART')

    TRAINING_DATA = get_training_data()
    # Start the training
    optimizer = nlp.begin_training()

    # Iterate 15 times
    for itn in range(15):
        # Shuffle the data
        random.shuffle(TRAINING_DATA)
        losses = {}

        # Batch the examples and iterate over them
        for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
            for text, entities in batch:
                doc = nlp.make_doc(text)
                nlp.update([doc], [entities], sgd=optimizer, losses=losses)

            # texts = [text for text, entities in batch]
            # entity_offsets = [entities for text, entities in batch]

            # Update the model
            # nlp.update(texts, entity_offsets, sgd=optimizer, losses=losses)
        print(losses)

    nlp.to_disk('/home/viktor/Documents/avisaf-ner/example-model-1')
    print('Model saved')

    print('Testing')
    nlp = spacy.load('/home/viktor/Documents/avisaf-ner/example-mode-1')
    i = 0
    # test the trained model
    for text, _ in TRAINING_DATA:
        i += 1
        if i > 10:
            break
        else:
            doc = nlp(text)
            print("Entities: ", "\t{}".format([(ent.text, ent.label_) for ent in doc.ents]), sep='\n')
            # print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
