#!/usr/bin/env python3

import spacy
from spacy.displacy import serve
from sys import stderr

"""
def skusam():
    nlp = spacy.load('/home/viktor/Documents/avisaf-ner/example-model')
    sent = nlp.create_pipe('sentencizer')
    nlp.add_pipe(sent)
    TR_DATA = get_training_data()
    lst = []
    for text, _ in TR_DATA:
        doc = nlp(text)
        json_data = docs_to_json(doc)
        lst.append(json_data)

    with open('/home/viktor/Documents/avisaf-ner/dev-json', mode='a') as jf:
        json.dump(lst, jf)
"""


def obtain_text():
    return "The text which will be here. Barack Obama was a president of the United States of America."


if __name__ == '__main__':
    # TODO: a user should be able to either copy the text, or choose a txt file to be processed which will extract the text and identify the entities
    # extract the text
    text = obtain_text()

    # create new nlp object
    model = '/home/viktor/Documents/avisaf-ner/example-model'
    try:
        nlp = spacy.load(model)

        if not nlp.has_pipe(u'ner'):
            raise OSError

        # create doc object nlp(text)
        document = nlp(text)

        # identify entities
        entities = document.ents

        # print them using displacy renderer
        for ent in entities:
            print(ent.text, ent.label_)

        serve(document, style='ent')
    except OSError:
        print(f'The model \'{model}\' is not available or does not contain required components.', file=stderr)
        exit(1)
