#!/usr/bin/env python3

from spacy.matcher import PhraseMatcher
import spacy
import json
import os

if __name__ == '__main__':

    with open(os.path.expanduser('~/Documents/avisaf-ner/acft_parts_list.json'), mode='r') as file:
        WORDS = json.loads(file.read())

    nlp = spacy.load('en_core_web_sm')

    print(nlp.pipe_names)
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
        print(nlp.pipe_names)
    else:
        ner = nlp.get_pipe('ner')

    # doc = nlp('We pushed from the gate on time and taxied to the runway for departure.')
    doc = nlp('During the takeoff roll; a loose can of unopened Coca-Cola Zero rolled from behind the'
              ' captain\'s rudder pedals and stopped between the captain\'s left foot and the left '
              'rudder pedal.  Since I was the Pilot Monitoring; I was able to remove the object and '
              'the takeoff was continued.  Had I been the Pilot Flying; this event would have '
              'resulted in a rejected takeoff. This object was not from our crew and was lost/loose '
              'at some time prior to us beginning our pre-flight duties for this flight. This event '
              'would have been a much greater threat to safe operations had anything else irregular '
              'happened during the takeoff roll.')

    matcher = PhraseMatcher(nlp.vocab)
    patterns = list(nlp.pipe(WORDS))
    matcher.add("ACFT_PARTS", None, *patterns)

    matches = matcher(doc)

    for (match_id, start, end) in matches:
        print(match_id, start, end, doc[start:end])

    for entity in doc.ents:
        print(entity.text, entity.label_)

    # displacy.serve(doc, style="dep")
    # displacy.serve(doc, style="ent")