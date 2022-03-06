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
from argument_parser import parse_args
from pathlib import Path

# importing own modules
from training.new_entity_trainer import train_spacy_ner
from training.training_data_creator import ner_auto_annotation_handler, ner_man_annotation_handler
from classification.classifier import train_classification, launch_classification, evaluate_classification
from evaluation.ner_evaluator import evaluate_spacy_ner
from util.data_extractor import get_entities

logger = logging.getLogger("avisaf_logger")
logging.basicConfig(format=f"[%(levelname)s - %(asctime)s]: %(message)s")


def test_spacy_ner(
    model="en_core_web_md",
    text_path=None,
    cli_result: bool = False,
    visualize: bool = False,
    html_result_file: Path = None,
    port: int = 5000,
):
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
    """

    if text_path is None:
        # use sample text
        with open("sample_text.txt", mode="r") as sample_text_file:
            text = sample_text_file.read()
    else:
        # extract the text
        try:
            text_path = Path(text_path).resolve()
            text = text_path
            if text_path.exists():
                # in case the argument is the path to the file containing the text
                with text_path.open(mode="r") as file:
                    text = file.read()
        except OSError:
            # if the text is passed as argument
            text = text_path

    # create new nlp object
    if model.startswith("en_core_web"):
        logger.info("Using a default english language model!")
    try:
        # trying to load either the pre-trained spaCy model or a model in current directory
        nlp = spacy.load(model)
    except OSError:
        model_path = str(Path(model).resolve())
        nlp = spacy.load(model_path)

    if not nlp.has_pipe("ner"):
        logger.error(f"The model '{model}' is not available or does not contain required components.")
        return

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
            "colors": dict(
                zip(
                    ents_colors_dict.keys(),
                    map(lambda color_list: color_list[0], ents_colors_dict.values()),
                )
            ),
        }

        if html_result_file is None:
            displacy.serve(document, style="ent", options=options, port=port, host="localhost")
            return

        result_file_path = Path(html_result_file)
        result_file_path.touch(exist_ok=True)

        with result_file_path.open(mode="w") as file:
            html = displacy.render(document, style="ent", options=options)
            file.write(html)


def choose_action(args: Namespace):
    """Callback function which invokes the correct function with specified
    command-line arguments.

    :type args: Namespace
    :param args: argparse command-line arguments wrapped in Namespace object.
    """

    functions = {
        "train_ner": lambda: train_spacy_ner(
            iter_number=args.iterations,
            model=args.model,
            new_model_name=args.name,
            train_data_srcfiles=args.data,
            batch_size=args.batch_size
        ),
        "test_ner": lambda: test_spacy_ner(
            model=args.model,
            cli_result=args.print,
            visualize=args.render,
            text_path=args.text,
            html_result_file=args.save,
            port=args.port,
        ),
        "eval_ner": lambda: evaluate_spacy_ner(
            model=args.model,
            texts_file=args.texts
        ),
        "annotate_auto": lambda: ner_auto_annotation_handler(
            args.keys_file,
            args.label,
            model=args.model,
            training_src_file=args.data,
            extract_texts=args.extract,
            use_phrasematcher=args.p,
            save=args.save,
            save_to=args.save_to
        ),
        "annotate_man": lambda: ner_man_annotation_handler(
            labels_path=args.labels,
            file_path=args.texts_file,
            lines=args.lines,
            save=args.not_save,
            start_index=args.start_index,
        ),
        "classifier_train": lambda: train_classification(
            models_paths=args.model,
            texts_paths=args.paths,
            label=args.label,
            label_values=args.filter,
            algorithm=args.algorithm,
            normalization=args.normalize
        ),
        "classifier_process": lambda: launch_classification(
            model_path=args.model,
            text_paths=args.paths
        ),
        "classifier_eval": lambda: evaluate_classification(
            model_path=args.model,
            text_paths=args.paths,
            compare_baseline=args.compare_baseline,
            show_curves=args.show_curves,
        )
    }

    try:
        functions.get(args.dest, lambda: logger.error(f"Desired function \"{args.dest}\" is not supported."))()
    except AttributeError as ex:
        logger.error(ex.with_traceback(sys.exc_info()[0]))
    except OSError as e:
        logger.error(e.with_traceback(sys.exc_info()[2]))


def main():
    """Main function of the program. This function parses command-line arguments
    using argparse module before performing appropriate callback which actually
    executes desired operation.
    """
    main_parser, avail_parsers = parse_args()

    if len(sys.argv) <= 1:
        main_parser.print_help()
        return

    if len(sys.argv) == 2:
        subparser = avail_parsers.get(sys.argv[1])  # get correct subparser by its subcommand name
        subparser.print_help() if subparser else main_parser.print_help()
        return

    args = main_parser.parse_args()

    if not args.verbose or args.verbose == 0:
        logging.getLogger("avisaf_logger").setLevel(logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger("avisaf_logger").setLevel(logging.INFO)
    else:
        logging.getLogger("avisaf_logger").setLevel(logging.DEBUG)

    if args.dest == "ner_test":
        visualization_not_available = not args.print and not args.render and args.save is None
        if visualization_not_available:
            print("The output will not be visible without one of --print, --render or --save argument.\n")
            ner_tester = avail_parsers.get("ner_test")
            if ner_tester:
                ner_tester.print_help()

    choose_action(args)


if __name__ == "__main__":
    main()
