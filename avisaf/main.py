#!/usr/bin/env python3
"""Avisaf module is a tool for extracting and highlighting aviation related
terminology and expressions based on a Natural Language Processing library
spaCy. This program is able to highlight aviation related entities, train
new models using existing examples but also build new and improve existing
entity recognition models.
"""

import spacy
import spacy.displacy as displacy
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from colorama import Style, Fore
# importing own modules
from avisaf.training.new_entity_trainer import train_spacy_model
from avisaf.training.training_data_creator import annotate_auto, annotate_man
from avisaf.classification.classifier import launch_classification
from avisaf.util.data_extractor import get_entities

sample_text = ("Flight XXXX at FL340 in cruise flight; cleared direct to ZZZZZ intersection to join the XXXXX arrival "
               "to ZZZ and cleared to cross ZZZZZ1 at FL270. Just after top of descent in VNAV when the throttles "
               "powered back for descent a loud bang came from the left side of the aircraft followed by significant "
               "airframe vibration. No EICAS messages were observed at this time however a check of the engine synoptic"
               " revealed high vibration coming from the Number 2 Engine. I brought the Number 2 Throttle to idle but "
               "the vibration continued and severe damage was determined. We ran the severe damage checklist and "
               "secured the engine and then requested a slower speed from ATC to lessen the vibration and advised ATC. "
               "The slower speed made the vibration acceptable and the flight continued to descend on the arrival via "
               "ATC instructions. The FO was dispatched to the main deck to visually survey damage. He returned with "
               "pictures of obvious catastrophic damage of the Number 2 Engine and confirmed no visible damage to the "
               "leading edge or any other visible portion of the left side of the aircraft. The impending three engine "
               "approach; landing and possible go-around were talked about and briefed as well as the possibilities of "
               "leading and trailing edge flap malfunctions. A landing on Runway XXC followed and the aircraft was "
               "inspected by personnel before proceeding to the gate. After block in; inspection of the Number 2 "
               "revealed extensive damage.A mention of the exceptional level of competency and professionalism "
               "exhibited by FO [Name1] and FO [Name] is in order; their calm demeanor and practical thinking should be"
               " attributed with the safe termination of Flight XXXX!")


def test(model='en_core_web_md',
         text_path=None,
         cli_result: bool = False,
         visualize: bool = False,
         html_result_file: Path = None):
    """Function which executes entity extraction and processing. The function
    loads and creates spaCy Language model object responsible for Named Entity
    Recognition. The target text to have its entities recognized may be passed
    either as string argument or a path to the file containing the text. If any
    of the above options is used, than a sample text is used as example.

    :type model: str
    :param model: The string representation of a spaCy model. Either an existing
        pre-downloaded spaCy model or a path to a local directory., defaults to
        'en_core_web_md'
    :type text_path: str
    :param text_path: String representing either a path to the file with the
        text to be inspected or the text itself., defaults to None
    :type cli_result: bool
    :param cli_result: A flag which will cause the result to be printed to the
        stdout., defaults to False
    :type visualize: bool
    :param visualize: A flag which will use the spaCy visualizer in order to
        render the result and show it in the browser., defaults to False
    :type html_result_file: Path
    :param html_result_file: The file path to the file where the result rendered
        by spaCy visualizer tool will be saved. The file will be created if it
        does not exist yet., defaults to None

    :return: The exit code of the function.
    :rtype: int
    """

    if text_path is None:
        # use sample text
        text = sample_text
    else:
        # extract the text
        try:
            text_path = Path(text_path).resolve()
            if text_path.exists():
                # in case the argument is the path to the file containing the text
                with text_path.open(mode='r') as file:
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

    except OSError as ex:
        print(ex)
        print(f'The model \'{model}\' is not available or does not contain required components.', file=sys.stderr)
        return 1

    # create doc object nlp(text)
    document = nlp(text)

    # identify entities
    if cli_result:
        result_string = ""
        for token in document:
            if token.ent_type_ == "":
                result_string += f"{token.text} "
            else:
                colors = {
                    "AIRPLANE": Fore.LIGHTGREEN_EX,
                    "CREW": Fore.LIGHTYELLOW_EX,
                    "AIRPORT_TERM": Fore.MAGENTA,
                    "FLIGHT_PHASE": Fore.LIGHTRED_EX,
                    "AVIATION_TERM": Fore.BLUE,
                    "NAV_WAYPOINT": Fore.LIGHTWHITE_EX,
                    "ALTITUDE": Fore.CYAN,
                    "WEATHER": Fore.LIGHTCYAN_EX,
                    "ABBREVIATION": Fore.RED
                }
                color = colors.get(token.ent_type_)
                result_string += f"{color}[{token.text}: {token.ent_type_}]{Style.RESET_ALL} "
        print(result_string)

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
            result_file_path = Path(html_result_file)
            result_file_path.touch(exist_ok=True)

            with result_file_path.open(mode='w') as file:
                html = displacy.render(document, style='ent', options=options)
                file.write(html)
        else:
            displacy.serve(document, style='ent', options=options)

    return 0


def choose_action(args: Namespace):
    """Callback function which invokes the correct function with specified
    command-line arguments.

    :type args: Namespace
    :param args: argparse command-line arguments wrapped in Namespace object.

    :return: The exit code of called function.
    :rtype: int
    """

    functions = {
        'train_ner': lambda: train_spacy_model(
            iter_number=args.iterations,
            model=args.model,
            new_model_name=args.name,
            tr_data_srcfile=Path(args.data),
            verbose=args.verbose
        ),
        'test_ner': lambda: test(
            model=args.model,
            cli_result=args.print,
            visualize=args.render,
            text_path=args.text,
            html_result_file=args.save
        ),
        'annotate_auto': lambda: annotate_auto(
            Path(args.keys_file),
            args.label,
            model=args.model,
            training_src_file=args.data,
            extract_texts=args.extract,
            use_phrasematcher=args.p,
            save=args.save,
            verbose=args.verbose
        ),
        'annotate_man': lambda: annotate_man(
            labels_path=Path(args.labels),
            file_path=Path(args.texts_file),
            lines=args.lines,
            save=args.not_save,
            start_index=args.start_index
        ),
        'classifier': lambda: launch_classification(
            label=args.label,
            texts_paths=args.paths,
            label_filter=args.filter,
            algorithm=args.algorithm,
            normalize=args.normalize,
            mode=args.mode,
            models_dir_paths=args.model,
            plot=args.plot
        )
    }

    try:
        func = functions.get(args.action)
        func()
        return 0
    except AttributeError as ex:
        print(ex.with_traceback(sys.exc_info()[0]), file=sys.stderr)
        print('No action is to be invoked.', file=sys.stderr)
        return 1
    except OSError as e:
        print(e.with_traceback(sys.exc_info()[2]), file=sys.stderr)
        return 1


def main():
    """Main function of the program. This function parses command-line arguments
    using argparse module before performing appropriate callback which actually
    executes desired operation.

    :return: The function returns the exit code of a sub-function. Any non-zero
             exit code means that the operation did not end successfully.
    :rtype: int
    """
    args = ArgumentParser(description='Named entity recognizer for aviation safety reports.')
    subparser = args.add_subparsers(help='Possible actions to perform.')

    # train subcommand and its arguments
    # ========================================================================================
    arg_train = subparser.add_parser(
        'train_ner',
        help='Train a new NLP NER model.',
        description='Command for training new/updating entities.'
    )
    arg_train.set_defaults(action='train_ner')
    arg_train.add_argument(
        '-d', '--data',
        metavar='PATH',
        help='File path to the file with annotated training data.',
        default=Path('data_files', 'training_data', 'annotated_data_part_01.json'),
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
    # ========================================================================================
    arg_test = subparser.add_parser(
        'test_ner',
        help='Test a selected model.',
        description='Command used for testing the entity recognition on given text.'
    )
    arg_test.set_defaults(action='test_ner')
    arg_test.add_argument(
        '-m', '--model',
        metavar='PATH/MODEL',
        default='en_core_web_md',
        required=True,
        help='File path to an existing spaCy model or existing spaCy model name for NER.'
    )
    group = arg_test.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-p', '--print',
        action='store_true',
        help='Print the result on the screen.'
    )
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
    # ========================================================================================
    arg_autobuild = subparser.add_parser(
        'autobuild',
        help='Automatic annotation tool for new training dataset creation.',
        description='Automatic annotation tool for new training dataset creation.'
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
        action='store_true',
        help='Flag indicating that spaCy\'s PhraseMatcher object should be used.'
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
    # ========================================================================================
    arg_manbuild = subparser.add_parser(
        'build',
        help='Manual annotation tool for new training dataset creation.',
        description='Manual annotation tool for new training dataset creation.'
    )
    arg_manbuild.set_defaults(action='annotate_man')
    arg_manbuild.add_argument(
        'texts_file',
        help='''The path to the file containing texts to be annotated. (Supports .csv/.json files).
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
        default=Path('entities_labels.json')
    )
    arg_manbuild.add_argument(
        '-s', '--start-index',
        type=int,
        help='The index of the first text to be annotated.',
        default=0
    )
    arg_manbuild.add_argument(
        '--not_save',
        action='store_false',
        help='Flag indicating whether the result of the annotation should NOT be saved.',
    )

    # classification module and its arguments
    # ========================================================================================
    arg_classifier = subparser.add_parser(
        'classifier',
        help='Train an ASRS reports classification model.'
    )
    arg_classifier.set_defaults(action='classifier')
    arg_classifier.add_argument(
        '--paths',
        nargs='+',
        help='Strings representing the paths to training data texts',
        default=[]
    )
    arg_classifier.add_argument(
        '--mode',
        choices={'train', 'dev', 'test'},
        default='test',
        help='Choose classifier operating mode (default test)'
    )
    arg_classifier.add_argument(
        '-l', '--label',
        help='The label of the column to be extracted from the documents (in format FirstLineLabel_SecondLineLabel)',
        default=None,
    )
    arg_classifier.add_argument(
        '-f', '--filter',
        nargs='*',
        help='Subset of the values present in the column given by the label',
        default=None
    )
    arg_classifier.add_argument(
        '-a', '--algorithm',
        default='mlp',
        help='The algorithm used for classification training.',
        choices={'knn', 'svm', 'mlp', 'forest', 'gauss', 'mnb', 'regression'}
    )
    arg_classifier.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize the distribution of classes in training data'
    )
    arg_classifier.add_argument(
        '--plot',
        action='store_true',
        help='Show AUC for each of selected models'
    )
    arg_classifier.add_argument(
        '-m', '--model',
        default=None,
        nargs='+',
        help='Trained model(s) to use (at least one is required)',
    )

    if len(sys.argv) <= 1:
        args.print_help()
        return 0

    if len(sys.argv) == 2:
        helpers = {
            'test': arg_test.print_help,
            'train': arg_train.print_help,
            'autobuild': arg_autobuild.print_help,
            'build': arg_manbuild.print_help,
            'train_classifier': arg_classifier.print_help
        }

        help_function = helpers.get(sys.argv[1])
        if help_function is not None:
            help_function()
        else:
            args.print_help()
        return 0
    else:
        parsed = args.parse_args()
        if parsed.action == 'test_ner' and not parsed.print and not parsed.render and parsed.save is None:
            print("The output will not be visible without one of --print, --render or --save argument.\n", file=sys.stderr)
            arg_test.print_help()
            return 1
        exit_code = choose_action(parsed)
        return exit_code


if __name__ == '__main__':
    retval = main()
    sys.exit(retval)
