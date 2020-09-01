#!/usr/bin/env python3

import spacy
from spacy.displacy import serve
from sys import stderr


def obtain_text():
    t = ("Flight XXXX at FL340 in cruise flight; cleared direct to ZZZZZ intersection to join the XXXXX arrival to "
         "ZZZ and cleared to cross ZZZZZ1 at FL270. Just after top of descent in VNAV when the throttles powered "
         "back for descent a loud bang came from the left side of the aircraft followed by significant airframe "
         "vibration. No EICAS messages were observed at this time however a check of the engine synoptic revealed "
         "high vibration coming from the Number 2 Engine. I brought the Number 2 Throttle to idle but the vibration "
         "continued and severe damage was determined. We ran the severe damage checklist and secured the engine and "
         "then requested a slower speed from ATC to lessen the vibration and advised ATC. The slower speed made the "
         "vibration acceptable and the flight continued to descend on the arrival via ATC instructions. The FO was "
         "dispatched to the main deck to visually survey damage. He returned with pictures of obvious catastrophic "
         "damage of the Number 2 Engine and confirmed no visible damage to the leading edge or any other visible "
         "portion of the left side of the aircraft. The impending three engine approach; landing and possible "
         "go-around were talked about and briefed as well as the possibilities of leading and trailing edge flap "
         "malfunctions. A landing on Runway XXC followed and the aircraft was inspected by personnel before "
         "proceeding to the gate. After block in; inspection of the Number 2 revealed extensive damage.A mention of "
         "the exceptional level of competency and professionalism exhibited by FO [Name1] and FO [Name] is in order;"
         " their calm demeanor and practical thinking should be attributed with the safe termination of Flight XXXX!"
         )

    return ('During the takeoff roll; a loose can of unopened Coca-Cola Zero rolled from behind the'
            ' captain\'s rudder pedals and stopped between the captain\'s left foot and the left '
            'rudder pedal.  Since I was the Pilot Monitoring; I was able to remove the object and '
            'the takeoff was continued.  Had I been the Pilot Flying; this event would have '
            'resulted in a rejected takeoff. This object was not from our crew and was lost/loose '
            'at some time prior to us beginning our pre-flight duties for this flight. This event '
            'would have been a much greater threat to safe operations had anything else irregular '
            'happened during the takeoff roll.')


if __name__ == '__main__':
    # TODO: a user should be able to either copy the text, or choose a txt file to be processed which will extract the text and identify the entities
    # TODO: a user should be able to use own spaCy model defined as name in the parameter --model=str

    # extract the text
    text = obtain_text()

    # create new nlp object
    model_name = '/home/viktor/Documents/avisaf_ner/models/retrained-model'
    try:
        nlp = spacy.load(model_name)

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
        print(f'The model \'{model_name}\' is not available or does not contain required components.', file=stderr)
        exit(1)