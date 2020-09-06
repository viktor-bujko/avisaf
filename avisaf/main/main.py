#!/usr/bin/env python3

import spacy
import spacy.displacy as displacy
import sys
import os
from argparse import ArgumentParser

PROJECT_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(PROJECT_ROOT_PATH, 'avisaf'))
from trainer.new_entity_trainer import train_spaCy_model
from trainer.training_data_creator import annotate_auto, annotate_man


def obtain_text():
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


def test(model='en_core_web_md', cli_result=False, visualize=False):
    # TODO: a user should be able to either copy the text, or choose a txt file to be processed which will extract the 
    #       text and identify the entities
    # TODO: a user should be able to use own spaCy model defined as name in the parameter --model=str

    # extract the text
    text = obtain_text()

    try:
        # create new nlp object
        if model.startswith('en_core_web'):
            print('Using a default english language model!')
        nlp = spacy.load(model)

        if not nlp.has_pipe(u'ner'):
            raise OSError

        # create doc object nlp(text)
        document = nlp(text)

        # identify entities
        entities = document.ents

        # print them using displacy renderer
        for ent in entities:
            print(f'{ent.text}\t{ent.label_}', flush=cli_result)

        if visualize:
            displacy.serve(document, style='ent')

    except OSError:
        print(f'The model \'{model}\' is not available or does not contain required components.', file=sys.stderr)
        exit(1)


def choose_action(args):
    """
    Callback function which invokes the correct function with specified
    CLI arguments.
    :param args: parseargs command-line arguments.
    :return:    Exit code.
    """

    FUNCTIONS = {
        'train': lambda: train_spaCy_model(iter_number=args.iterations, model=args.model,
                                           new_model_name=args.name, train_data_srcfile=args.data,
                                           verbose=args.verbose),

        'test': lambda: test(model=args.model, cli_result=args.print, visualize=args.show),

        'annotate_auto': lambda: annotate_auto(args.keys_file, args.label,
                                               model=args.model, tr_src_file=args.data,
                                               extract_texts=args.extract, use_phrasematcher=args.p,
                                               save=args.save, verbose=args.verbose),

        'annotate_man': lambda: annotate_man(labels_path=args.labels, file_path=args.texts_file,
                                             lines=args.lines, save=args.save,
                                             start_index=args.start_index)
    }

    try:
        func = FUNCTIONS.get(args.action)
        func()
        return 0
    except AttributeError:
        print('No action is to be invoked.', file=sys.stderr)
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
    arg_train.set_defaults(action='train_spaCy_model')
    arg_train.add_argument(
        '-i', '--iterations',
        metavar='INT',
        type=int,
        default=20,
        help='The number of iterations to perform for entity training.'
    )
    arg_train.add_argument(
        '-m', '--model',
        metavar='PATH',
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
        '-d', '--data',
        metavar='PATH',
        help='File path to the file with annotated training data.',
        default=os.path.join(PROJECT_ROOT_PATH, 'data_files/auto_annotated_data.json')
    )
    arg_train.add_argument(
        '-v', '--verbose',
        type=bool,
        metavar='BOOL',
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
        metavar='PATH',
        default='en_core_web_md',
        help='File path to an existing spaCy model or existing spaCy model name for NER.'
    )
    arg_test.add_argument(
        '-s', '--show',
        type=bool,
        default=False,
        help='A flag to indicate whether a visualization tool should be started.'
    )
    arg_test.add_argument(
        '-p', '--print',
        type=bool,
        default=False,
        help='Print the result on the screen.'
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
        '-m', '--model',
        type=str,
        help='File path to an existing spaCy model or existing spaCy model name.',
        default='en_core_web_md'
    )
    arg_autobuild.add_argument(
        '-d', '--data',
        type=str,
        help='Training data source file path.',
        default=None
    )
    arg_autobuild.add_argument(
        '-e', '--extract',
        type=bool,
        help='Flag indicating that text extraction should take place.',
        default=False
    )
    arg_autobuild.add_argument(
        '-p',
        type=bool,
        help='Flag indicating that spaCy\'s PhraseMatcher object should be used.',
        default=True
    )
    arg_autobuild.add_argument(
        '-s', '--save',
        type=bool,
        help='Flag indicating that the result should be saved. Requires the -d/--data argument.',
        default=False
    )
    arg_autobuild.add_argument(
        '-v', '--verbose',
        type=bool,
        help='Flag indicating verbose printing.',
        default=False
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
        type=bool,
        help='Flag indicating whether the result of the annotation should be saved.',
        default=False
    )

    args = args.parse_args()
    exit_code = choose_action(args)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
