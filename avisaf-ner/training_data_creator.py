#!/usr/bin/env python3

from spacy.matcher import PhraseMatcher
from util.file_choice import extract_narratives
from util.indexing import get_spans_indexes
import spacy
import json
import os


def create_tr_data():
    TRAINING_DATA = []

    # create first rule-based keys corresponding to acft parts
    with open(os.path.expanduser('~/Documents/avisaf-ner/acft_parts_list.json'), mode='r') as keys_file:
        # replace fixed file path with path argument for dictionary
        KEYS = json.loads(keys_file.read())  # patterns to be matched

    # get testing texts;
    TEXTS = list(extract_narratives(lines=20))

    '''TEXTS = [('During the takeoff roll; a loose can of unopened Coca-Cola Zero rolled from behind the'
              ' captain\'s rudder pedals and stopped between the captain\'s left foot and the left '
              'rudder pedal.  Since I was the Pilot Monitoring; I was able to remove the object and '
              'the takeoff was continued.  Had I been the Pilot Flying; this event would have '
              'resulted in a rejected takeoff. This object was not from our crew and was lost/loose '
              'at some time prior to us beginning our pre-flight duties for this flight. This event '
              'would have been a much greater threat to safe operations had anything else irregular '
              'happened during the takeoff roll.'),
             'We pushed from the gate on time and taxied to the runway for departure.'
             ]'''

    # create NLP analyzer object
    nlp = spacy.load('en_core_web_sm')
    # create PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab)

    # process the keys and store their values in the patterns list
    patterns = list(nlp.pipe(KEYS))
    # add all patterns to the matcher
    matcher.add('ACFT_PART', None, *patterns)
    i = 0

    for doc in nlp.pipe(TEXTS[1101:]):
        matched_spans = [doc[start:end] for _, start, end in matcher(doc)]
        entities = [(span.start_char, span.end_char, 'ACFT_PART') for span in matched_spans]
        tr_example = (doc.text, {"entities": entities})
        TRAINING_DATA.append(tr_example)

    print(*TRAINING_DATA, sep='\n')


def annotate_texts(labels):
    TEXTS = list(extract_narratives(lines=2))
    result = []
    for text in TEXTS:
        ent_labels = []
        print(text)
        words = input("Write all words you want to annotate: ")
        spans = [word.strip() for word in words.split(',')]

        found_occurs = get_spans_indexes(text, spans)
        for occur_dict in found_occurs:
            key = list(occur_dict.keys())[0]
            matches = occur_dict[key]
            label = input(f'Label \'{key}\' with one of the following {labels}: ').upper()
            ent_labels += [(start, end, label) for start, end in matches]
        result.append((text, {"entities": ent_labels}))

    return result


if __name__ == '__main__':
    annotated = annotate_texts(['ACFT_PART', 'CREW', 'APT_TERM', 'FLIGHT_TERM', 'AV_TERM'])
    print(annotated)
