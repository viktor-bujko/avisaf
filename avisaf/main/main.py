#!/usr/bin/env python3

import spacy
import spacy.displacy as displacy
import sys
import os
from argparse import ArgumentParser
from pathlib import Path

SOURCES_ROOT_PATH = Path(__file__).parent.parent.resolve()
PROJECT_ROOT_PATH = SOURCES_ROOT_PATH.parent.resolve()
sys.path.append(str(SOURCES_ROOT_PATH))

from trainer.new_entity_trainer import train_spaCy_model
from trainer.training_data_creator import annotate_auto, annotate_man
from util.data_extractor import get_entities


def get_sample_text():
    ex =   ("Departed ramp and taxied to Runway 4. Arrived at runway 4 run-up area and performed pre-flight run-up. All"
            " indications were satisfactory and within limitations. Taxied out of run-up area to the hold short line of"
            " Runway 4. Received takeoff clearance from Runway 4 and proceeded to taxi onto runway at which point full"
            " power was added and a takeoff was initiated. Another check of the instruments was done as required and"
            " all were within limitations; RPMs were 2400; and airspeed was rising steadily. After rotation at "
            "approximately 200 feet a loud bang came from the engine compartment that sounded like the engine "
            "backfiring but normal operation continued. Upon reaching approximately 400 feet engine power loss began. "
            "Engine power dropped by about 400-500 RPMs to approximately 2000 RPMs. After the initial drop; RPMs rose "
            "by about 200 RPMs to 2200 RPMs. However; following the rise; the engine RPMs dropped in and out ranging "
            "from 200 RPM drops to 1000 RPM drops. At this point a sufficient climb was unable to be maintained due to "
            "loss of power. Tower was contacted and a request to return to the opposite direction runway (Runway 22) "
            "was made. Tower cleared all traffic from the runway and gave priority handling to us. At that point a "
            "landing was made on Runway 22. Due to excessive braking from landing on a shortened runway (only about 50%"
            " of runway was remaining at touchdown; about 2;000 feet) and a tailwind of 12 knots gusting to 19 knots "
            "the right main gear tire became worn but did not blow out. There was however a large flat spot on the tire"
            ". After making the landing; turned off the runway and returned to ramp where a secondary run-up was "
            "performed. The only noticeable problem was that when checking the right magneto a popping noise was made "
            "followed by a drop of 200-300 RPMs; but would then rise and steadily maintain an RPM setting within "
            "limitations (approximately 100 RPMs below RPM setting for run-up). Incident was reported to maintenance "
            "for further review. No damage was done to the aircraft and the instructor pilot did all flying after the "
            "initial engine power loss was observed. Student pilot and observing passenger were onboard the aircraft.")

    return ("Flight XXXX at FL340 in cruise flight; cleared direct to ZZZZZ intersection to join the XXXXX arrival to "
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


def test(model='en_core_web_md',
         text_path=None,
         cli_result: bool = False,
         visualize: bool = False,
         html_result_file: Path = None):
    """

    :type model:            str
    :param model:
    :type text_path:        Path
    :param text_path:
    :type cli_result:       bool
    :param cli_result:
    :type visualize:        bool
    :param visualize:
    :type html_result_file: Path
    :param html_result_file:
    :return:
    """

    if text_path is None:
        # use sample text
        # text = input("Please enter the text: \n")
        text = get_sample_text()
    else:
        # extract the text
        try:
            text_Path = Path(text_path).resolve()
            if text_Path.exists():
                # in case the argument is the path to the file containing the text
                with text_Path.open(mode='r') as file:
                    text = file.read()
            else:
                raise OSError
        except OSError:
            # if the text is passed as argument
            text = text_path

    try:
        # create new nlp object
        if model.startswith('en_core_web'):
            print('Using a default english language model!', file=sys.stderr)
        try:
            # trying to load either the pre-trained spaCy model or a model in current directory
            nlp = spacy.load(model)
        except OSError:
            model_path = str(Path(model).resolve())
            nlp = spacy.load(model_path)

        if not nlp.has_pipe(u'ner'):
            raise OSError

    except OSError:
        print(f'The model \'{model}\' is not available or does not contain required components.', file=sys.stderr)
        nlp = None
        exit(1)

    # create doc object nlp(text)
    document = nlp(text)

    # identify entities
    entities = document.ents

    longest_entity = max([len(entity.text) for entity in entities])
    # print them using displacy renderer
    for ent in entities:
        dist = longest_entity - len(ent.text) + 4
        print(f'{ent.text}{" " * dist}{ent.label_}', flush=cli_result)

    if html_result_file is not None or visualize:
        colors = {
            "AIRPLANE": "#ACECD5",
            "CREW": "#FFF9AA",
            "AIRPORT_TERM": "#FFD5B8",
            "FLIGHT_PHASE": "#FFB9B3",
            "AVIATION_TERM": "#A5C8E4",
            "NAV_WAYPOINT": "#FF6961",
            "ALTITUDE": "#988270",
            "WEATHER": "#BE9B7B",
            "ABBREVIATION": "#FFF4E6"
        }
        options = {
            "ents": get_entities(),
            "colors": colors
        }
        if html_result_file is not None:
            result_file_Path = Path(html_result_file)
            result_file_Path.touch(exist_ok=True)

            with result_file_Path.open(mode='w') as file:
                html = displacy.render(document, style='ent', options=options)
                file.write(html)
        else:
            displacy.serve(document, style='ent', options=options)


def choose_action(args):
    """
    Callback function which invokes the correct function with specified
    CLI arguments.
    :param args: parseargs command-line arguments.
    :return:    Exit code.
    """

    FUNCTIONS = {
        'train': lambda: train_spaCy_model(iter_number=args.iterations, model=args.model,
                                           new_model_name=args.name, tr_data_srcfile=Path(args.data),
                                           verbose=args.verbose),

        'test': lambda: test(model=args.model, cli_result=args.print,
                             visualize=args.render, text_path=args.text,
                             html_result_file=args.save),

        'annotate_auto': lambda: annotate_auto(Path(args.keys_file), args.label,
                                               model=args.model, tr_src_file=Path(args.data),
                                               extract_texts=args.extract, use_phrasematcher=args.p,
                                               save=args.save, verbose=args.verbose),

        'annotate_man': lambda: annotate_man(labels_path=Path(args.labels), file_path=Path(args.texts_file),
                                             lines=args.lines, save=args.save,
                                             start_index=args.start_index)
    }

    try:
        func = FUNCTIONS.get(args.action)
        func()
        return 0
    except AttributeError as ex:
        print(ex, file=sys.stderr)
        print('No action is to be invoked.', file=sys.stderr)
        return 1
    except OSError as e:
        print(e)
        return 1


def main():

    args = ArgumentParser(description='Named entity recognizer for aviation safety reports.')
    # args.set_defaults(action=args.print_help())
    subparser = args.add_subparsers(help='Possible actions to perform.')

    # train subcommand and its arguments
    arg_train = subparser.add_parser(
        'train',
        help='Train a new NLP NER model.'
    )
    arg_train.set_defaults(action='train')
    arg_train.add_argument(
        '-d', '--data',
        metavar='PATH',
        help='File path to the file with annotated training data.',
        default=os.path.join(PROJECT_ROOT_PATH, 'data_files/auto_annotated_data.json'),
        required=True
    )
    arg_train.add_argument(
        '-i', '--iterations',
        metavar='INT',
        type=int,
        default=20,
        help='The number of iterations to perform for entity training.'
    )
    arg_train.add_argument(
        '-m', '--model',
        metavar='PATH/NAME',
        help='File path to an existing spaCy model or existing spaCy model name to be trained.',
        default=None
    )
    arg_train.add_argument(
        '-n', '--name',
        metavar='STRING',
        help='Name of the new model.',
        default=None
    )
    arg_train.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Flag for verbose printing.'
    )

    # test subcommand and its arguments
    arg_test = subparser.add_parser(
        'test',
        help='Test a selected model.'
    )
    arg_test.set_defaults(action='test')
    arg_test.add_argument(
        '-m', '--model',
        metavar='PATH/MODEL',
        default='en_core_web_md',
        help='File path to an existing spaCy model or existing spaCy model name for NER.'
    )
    arg_test.add_argument(
        '-p', '--print',
        action='store_false',
        help='Print the result on the screen.'
    )
    group = arg_test.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '-r', '--render',
        action='store_true',
        help='A flag to indicate whether a visualization tool should be started.',
    )
    group.add_argument(
        '-s', '--save',
        metavar='PATH',
        default=None,
        help='Save rendered html result into the file (will be created if does not exist).'
    )
    arg_test.add_argument(
        '-t', '--text',
        default=None,
        help='File path to the text which will have entities extracted. If None, sample text is used.'
    )

    # automatic training data builder and its arguments
    arg_autobuild = subparser.add_parser(
        'autobuild',
        help='Automatic annotation tool for new training dataset creation.'
    )
    arg_autobuild.set_defaults(action='annotate_auto')
    arg_autobuild.add_argument(
        'keys_file',
        help='Path to file with words to be matched.'
    )
    arg_autobuild.add_argument(
        'label',
        type=str,
        help='The text of the label of an entity.'
    )
    arg_autobuild.add_argument(
        '-d', '--data',
        type=str,
        help='Training data source file path.',
        default=None
    )
    arg_autobuild.add_argument(
        '-e', '--extract',
        action='store_true',
        help='Flag indicating that text extraction should take place.'
    )
    arg_autobuild.add_argument(
        '-m', '--model',
        type=str,
        help='File path to an existing spaCy model or existing spaCy model name.',
        default='en_core_web_md'
    )
    arg_autobuild.add_argument(
        '-p',
        action='store_false',
        help='Flag indicating that spaCy\'s PhraseMatcher object should NOT be used.'
    )
    arg_autobuild.add_argument(
        '-s', '--save',
        action='store_true',
        help='Flag indicating that the result should be saved. Requires the -d/--data argument.'
    )
    arg_autobuild.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Flag indicating verbose printing.',
    )

    # manual training data builder and its arguments
    arg_manbuild = subparser.add_parser(
        'build',
        help='Manual annotation tool for new training dataset creation.'
    )
    arg_manbuild.set_defaults(action='annotate_man')
    arg_manbuild.add_argument(
        'texts_file',
        # type=str,
        help='''The path to the file containing texts to be annotated. 
                If None, then a user can write own sentences and annotate them.''',
        default=None
    )
    arg_manbuild.add_argument(
        'lines',
        type=int,
        help='The number of texts to be annotated (1 text = 1 line).'
    )
    arg_manbuild.add_argument(
        '-l', '--labels',
        help='Path to the file containing entity labels used for annotation.',
        default=None
    )
    arg_manbuild.add_argument(
        '-s', '--start-index',
        type=int,
        help='The index of the first text to be annotated.',
        default=0
    )
    arg_manbuild.add_argument(
        '--save',
        action='store_true',
        help='Flag indicating whether the result of the annotation should be saved.',
    )

    args = args.parse_args()
    exit_code = choose_action(args)

    return exit_code


if __name__ == '__main__':
    # todo: add aviation terminology glossary for entity "aviation_term"
    sys.exit(main())
