#!/usr/bin/env python3
"""Avisaf module is a tool for extracting and highlighting aviation related
terminology and expressions based on a Natural Language Processing library
spaCy. This program is able to highlight aviation related entities, train
new models using existing examples but also build new and improve existing
entity recognition models.
"""
import logging

import spacy
import spacy.displacy as displacy
import sys
from argparse import Namespace
from .argument_parser import parse_args
from pathlib import Path
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
         html_result_file: Path = None,
         port: int = 5000):
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
    :type port: int
    :param port: The number of the port to be used for displaCy rendered
        visualization. Argument only used when visualize is true or html_result_file
        is defined.

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
        print(f'The model \'{model}\' is not available or does not contain required components.', file=sys.stderr)
        return 1

    # create doc object nlp(text)
    document = nlp(text)

    ents_colors_dict = get_entities()
    # identify entities
    if cli_result:

        def print_highlighted_entity(tkn):
            if not tkn.ent_type_:
                print(f"{tkn.text} ", end="")
            else:
                color = ents_colors_dict.get(tkn.ent_type_)[1]
                print(f"\x1b[{color}m[{tkn.text}: {tkn.ent_type_}]\033[0m ", end="")

        for token in document:
            print_highlighted_entity(token)
        print()  # to end the text

    if html_result_file is not None or visualize:
        options = {
            "ents": list(ents_colors_dict.keys()),
            "colors": dict(zip(
                ents_colors_dict.keys(),
                map(lambda color_list: color_list[0], ents_colors_dict.values())
            ))
        }

        if html_result_file is None:
            displacy.serve(document, style='ent', options=options, port=port, host="localhost")
            return

        result_file_path = Path(html_result_file)
        result_file_path.touch(exist_ok=True)

        with result_file_path.open(mode='w') as file:
            html = displacy.render(document, style='ent', options=options)
            file.write(html)

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
            iter_number=args.iterations, model=args.model,
            new_model_name=args.name, tr_data_srcfile=Path(args.data),
            verbose=args.verbose
        ),
        'test_ner': lambda: test(
            model=args.model, cli_result=args.print,
            visualize=args.render, text_path=args.text,
            html_result_file=args.save, port=args.port
        ),
        'annotate_auto': lambda: annotate_auto(
            Path(args.keys_file), args.label,
            model=args.model, training_src_file=args.data,
            extract_texts=args.extract, use_phrasematcher=args.p,
            save=args.save, verbose=args.verbose
        ),
        'annotate_man': lambda: annotate_man(
            labels_path=Path(args.labels), file_path=Path(args.texts_file),
            lines=args.lines, save=args.not_save,
            start_index=args.start_index
        ),
        'classifier': lambda: launch_classification(
            label=args.label, texts_paths=args.paths,
            label_filter=args.filter, algorithm=args.algorithm,
            normalize=args.normalize, mode=args.mode,
            models_dir_paths=args.model, plot=args.plot
        )
    }

    try:
        func = functions.get(args.dest)
        if func:
            func()
        else:
            print('No action is to be invoked.', file=sys.stderr)
            return 1
        return 0
    except AttributeError as ex:
        logging.debug(ex.with_traceback(sys.exc_info()[0]))
        return 1
    except OSError as e:
        logging.debug(e.with_traceback(sys.exc_info()[2]))
        return 1


def main() -> int:
    """Main function of the program. This function parses command-line arguments
    using argparse module before performing appropriate callback which actually
    executes desired operation.

    :return: The function returns the exit code of a sub-function. Any non-zero
             exit code means that the operation did not end successfully.
    :rtype: int
    """
    main_parser, avail_parsers = parse_args()

    if len(sys.argv) <= 1:
        main_parser.print_help()
        return 0

    if len(sys.argv) == 2:
        subparser = avail_parsers.get(sys.argv[1])  # get correct subparser by its subcommand name
        subparser.print_help() if subparser else main_parser.print_help()
        return 0

    args = main_parser.parse_args()

    visualization_not_available = not args.print and not args.render and args.save is None

    if args.dest == 'ner_test' and visualization_not_available:
        print("The output will not be visible without one of --print, --render or --save argument.\n", file=sys.stderr)
        ner_tester = avail_parsers.get('ner_test')
        if ner_tester:
            ner_tester.print_help()
        return 1
    exit_code = choose_action(args)
    return exit_code


if __name__ == '__main__':
    retval = main()
    sys.exit(retval)
