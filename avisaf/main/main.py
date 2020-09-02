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

    return ["[We had a] left engine compressor stall just prior to level off at FL330. Left EPR was 1.97; right EPR was"
            " 2.08 left N2 was 87% right N2 was 92% EGTs were equal and stable. Smoke appeared in cockpit and cabin "
            "almost immediately. Smoke dissipated after approximately 5 minutes. Consulted and complied with QRH; "
            "contacted Dispatch; declared emergency with ATC; and diverted. Engine was operated throughout descent; "
            "approach and landing. ARFF met aircraft upon arrival and advised everything looked normal. Center was very"
            " helpful and professional. Approach assigned us to fly an ILS which was NOTAMed out of service; when I "
            "questioned them about it they assigned us the other ILS approach which we flew and landed on "
            "uneventfully.",
            "While enroute RPLC; Manila Control failed to switch us to Hong Kong control at SABNO. We began attempting "
            "contact with Hong Kong and were then contacted on guard by another company flight to contact Hong Kong on "
            "their active frequency. We switched to this frequency and regained contact with Control within 8 minutes "
            "from SABNO. The Controller seemed satisfied with our quick response to the call on guard and gave us "
            "vectors. We discussed techniques for better FIR boundary awareness; including inserting the point in the "
            "fix page so that it is more prominent on the ND; and the need to be proactive in switching to the next "
            "Control despite release from the previous Controller. This might include using another radio to make such "
            "contact although in this case leaving #2 on guard allowed us to receive the guard call from Company "
            "aircraft.",
            "We arrived at the aircraft in the morning and performed a detailed inspection. All the gear pins and the "
            "ADG safety pin were in the storage closet. We taxied out and took off from Runway 14. At positive rate FO "
            "called 'gear up' and we received a gear disagree warning and an indication that the left main gear was "
            "still down which was confirmed by the tower. I performed the QRH and notified ATC; Dispatch and "
            "Maintenance. We returned to land as directed by the QRH. When we arrived at the gate I inspected the left "
            "main gear and found a gear pin installed. I had no reason to suspect a pin was missed on the walk around "
            "because there were 4 in the storage closet. First of all; we should have done a more thorough detailed "
            "inspection. Secondly; If Maintenance installs pins and gives the A/C an Airworthiness signoff; they should"
            " be sure to pull the pins; maybe a checklist should be implemented for Maintenance if one is not already "
            "in use. Lastly; The Company should mandate that if a pin is installed by anyone on our A/C; it should be "
            "required to have a bright; clean flag.",
            "After doing some airwork; I decided to land at [a nearby airport]. I listened to ATIS; which specified "
            "Runway 26 in use; and made an initial call up to [the airport] Tower about 13 NM east of the airport. "
            "Tower cleared me to make a straight-in and to report at 4 NM; at or above 1;500 FT (their standard arrival"
            " instruction). On a straight-in for Runway 26; at about 6-8 NM from the airport; I heard a Cherokee "
            "report; also east of the airport; for a straight-in. The Tower instructed me to follow the Cherokee."
            " I made visual contact with the Cherokee about 1;000 FT below me on a southerly heading south of the "
            "final approach course; at a virtual right angle to both me and the runway. The Tower instructed the "
            "Cherokee to turn inbound at which point I was surprised to observe the aircraft make a LEFT turn; away "
            "from the airport. This placed him traveling opposite direction to the airport and to my flight path; which"
            " meant he almost immediately went out of sight behind and below me. I notified the Tower that the Cherokee"
            " had turned and was now behind me. The Tower acknowledged and transmitted 'landing clearance cancelled';"
            " I thought to the Cherokee. I continued inbound on a straight-in; assuming the Cherokee was now "
            "maneuvering to my six o clock and presumably following me. As I neared the 4 NM reporting point; I heard "
            "the tower call an aircraft further out on the approach; telling him he was 'number 3 behind a Cherokee and"
            " a Mooney.' I saw the Mooney ahead of me; crossing the threshold; which I assumed made me number two. "
            "Assuming the Tower had misidentified me as a Cherokee; I called in as 'Cessna X; 3.8 miles out'. "
            "The Tower asked me to IDENT; which I did. On about a two-mile final; I heard a transmission from "
            "another aircraft; 'There's a Cessna next to me.' Looking around; I saw a Cherokee about 800 FT away to my "
            "right; on a converging final. I immediately turned and climbed to the left and transmitted that I was"
            " breaking off the approach. The Tower instructed me to overfly the runway and turn to the left downwind"
            " at the departure end. I did; after first ensuring that the landing Cherokee was below me. I overflew the"
            " runway at pattern altitude; turned left to the downwind when instructed by the Tower and made a normal "
            "landing. HUMAN FACTORS/LESSONS LEARNED: When the Cherokee unexpectedly turned away from the airport and "
            "passed behind me; I reported it to the Tower; and their acknowledgement (which I thought canceled the "
            "landing instructions given to the Cherokee) allowed me to assume the Tower Controller would manage "
            "separation and sequencing; I allowed myself to become complacent that the traffic was being sequenced "
            "behind me; out of sight. When the Tower misidentified me as a Cherokee to following traffic; I assumed "
            "it was a simple mix-up of aircraft type; a not-uncommon occurrence. I called in to correct my type; and "
            "when the Tower had me IDENT; I again assumed that the Controller was seeing the big picture and taking "
            "care of my sequencing and separation. I now think that the controller was actually talking to the Cherokee"
            " who had apparently maneuvered behind me and was converging on me from behind; how he didn't see me till "
            "the last minute I'll never know; but he may well be thinking the same thing about me. The knowledge that "
            "there was a Cherokee behind me; coupled with the Controller (I thought) mistakenly calling me a Cherokee;"
            " should have tipped me off that there was a potential for dangerous confusion here. I should have been "
            "more proactive about clearing up the confusion with the Controller; rather than assuming that an Ident "
            "would resolve everything. This is the closest I have come to another aircraft; and it occurs to me that "
            "it was the classic scenario for a mid-air: VFR conditions; in the pattern at a Tower-controlled airport. "
            "It was a powerful reminder that the responsibility for making see-and-avoid work rests with nobody but "
            "the PIC; not the guy at the scope and not the guy in the other airplane.",
            "I was climbing to 8;500 FT approximately 7 NM north northeast of [the nearest airport] when the engine "
            "quit at approximately 6;500 FT. I had noticed the fuel pressure dropping right before the engine quit. "
            "The fuel gauge is not particularly accurate but I knew that the left tank was very low. The tank had run "
            "dry which was not a great surprise. As soon as the engine stopped; I turned the aux fuel pump on and "
            "switched to the center main tank. The engine did not start which did surprise me. I pushed over to "
            "establish best glide speed and turned towards [the nearest airport] which was close but still far enough "
            "that I would need to head there directly. I double checked that the aux pump was on; I moved the mixture "
            "to rich; and I visually checked that the fuel selector was on the center tank. I tried adjusting the "
            "throttle and carb heat; but to no avail. I also check the mags; but given how the engine just wound down; "
            "it seemed unlikely that it was an ignition problem. I had been monitoring Approach. So I contacted the "
            "Controller and declared an emergency. I told him that I was about 7 NM north; passing through 5;500 FT; "
            "had lost my engine; was squawking 7700 and there was only one sole on board. He cleared me for Runway 18. "
            "Given that I was trying to figure out why the engine would not restart; I was not sure which runway was "
            "18. I told the Controller that I was heading to the runway on my right which was the closer of the two. He"
            " confirmed that it was 18 and cleared me in. There still was no fuel pressure even with the aux pump on. "
            "So I turned the fuel selector to the right tank and the engine started up within a second or two. I "
            "leveled off as it spun up and then started to climb to regain altitude in case it failed again. Although;"
            " I was pretty sure I had resolved the problem. I switched the selector back to the center main tank and "
            "felt the positive tick of the indentation. The engine continued to run. I called the Controller to "
            "terminate the emergency and let him know all was well.When I originally switched to the center tank; I "
            "must not have fully engaged the selector. When I ran through the checklist I should have physically "
            "checked the selector rather than just visually confirmed it was on the correct tank. If I had wiggled "
            "the selector; I probably would have moved it into the operable position.",
            "I observed Gulfstream V begin an early turn on the SJC9 departure procedure from the CD/FD/CIC position "
            "when the Local Controller spoke up and brought it to the attention of everyone in the Tower cab. "
            "Gulfstream V climbed Northbound directly toward CRJ900; the preceding departing aircraft. The Local "
            "Controller notified the TOGA (Departure Sector) Controller over the landline after the early turn was "
            "observed; without response. The two aircraft passed very close to each other because of the failure of "
            "Gulfstream V to comply with the SJC9 departure procedure. Vigilance and immediate coordination when "
            "aircraft are observed conducting unexpected flight operations that jeopardize safety of flight.",
            "I was training an advanced trainee on the final position. We were getting busy and our feeder controllers"
            " were giving a somewhat aggressive feed including leaving aircraft on a new arrival. When this happened;"
            " my trainee started to try and sort out some problems with the feed he was getting; forgot to turn "
            "Aircraft Y to final off the base leg. I did not realize that Aircraft Y didn't get the turn to final; "
            "resulting in a loss of separation between him and Aircraft X on the straight in to a parallel runway. I "
            "would recommend people think before leaving aircraft on this new arrival during busy sessions."
            ]


if __name__ == '__main__':
    # TODO: a user should be able to either copy the text, or choose a txt file to be processed which will extract the text and identify the entities
    # TODO: a user should be able to use own spaCy model defined as name in the parameter --model=str

    # extract the text
    text = obtain_text()

    # create new nlp object
    model_name = '/home/viktor/Documents/avisaf_ner/models/auto-generated-data-model-1'
    try:
        nlp = spacy.load(model_name)

        if not nlp.has_pipe(u'ner'):
            raise OSError

        # create doc object nlp(text)

        # document = nlp(text)

        for document in nlp.pipe(text):
            # identify entities
            entities = document.ents

            # print them using displacy renderer
            for ent in entities:
                print(ent.text, ent.label_)

            serve(document, style='ent')
            print('---------------------------------')
    except OSError:
        print(f'The model \'{model_name}\' is not available or does not contain required components.', file=stderr)
        exit(1)
