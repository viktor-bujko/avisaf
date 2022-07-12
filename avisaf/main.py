#!/usr/bin/env python3
"""Avisaf module is a tool for extracting and highlighting aviation related
terminology and expressions based on a Natural Language Processing library
spaCy. This program is able to highlight aviation related entities, train
new models using existing examples but also build new and improve existing
entity recognition models.
"""
import logging
import sys
from argparse import Namespace
from .argument_parser import parse_args

# importing own modules
from .ner.predictor import process_ner
from .ner.trainer import train_ner
from .ner.annotator import auto_annotation_handler, manual_annotation_handler
from .ner.evaluator import evaluate_ner
from .text_classification.trainer import train_classification
from .text_classification.evaluator import evaluate_classification
from .text_classification.predictor_decoder import launch_classification

logger = logging.getLogger("avisaf_logger")


def choose_action(args: Namespace):
    """Callback function which invokes the correct function with specified
    command-line arguments.

    :type args: Namespace
    :param args: argparse command-line arguments wrapped in Namespace object.
    """

    functions = {
        "train_ner": lambda: train_ner(
            config_file_path=args.config_path,
            model=args.model,
            new_model_name=args.name,
            train_data_srcfiles=args.data,
        ),
        "process_ner": lambda: process_ner(
            model=args.model,
            cli_result=args.print,
            visualize=args.render,
            text_path=args.text_path,
            text=args.text,
            html_result_file=args.save,
            port=args.port,
        ),
        "eval_ner": lambda: evaluate_ner(
            model=args.model,
            texts_file=args.texts
        ),
        "annotate_auto": lambda: auto_annotation_handler(
            args.keys_file,
            args.label,
            model=args.model,
            training_src_file=args.data,
            extract_texts=args.extract,
            use_phrasematcher=args.p,
            save=args.save,
            save_to=args.save_to
        ),
        "annotate_man": lambda: manual_annotation_handler(
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
            normalization=args.normalize,
            set_default=args.set_default_class,
            vectorizer_type=args.vectorizer,
            params_overrides=args.params_overrides
        ),
        "classifier_process": lambda: launch_classification(
            model_path=args.model,
            text_path=args.text_path,
            text=args.text
        ),
        "classifier_eval": lambda: evaluate_classification(
            model_path=args.model,
            text_paths=args.paths,
            compare_baseline=args.compare_baseline,
            show_curves=args.show_curves
        )
    }

    try:
        functions.get(args.dest, lambda: logger.error(f"Desired function \"{args.dest}\" is not supported."))()
    except AttributeError as ex:
        logger.error(ex, exc_info=ex)
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

    logging.basicConfig(
        format=f"[%(levelname)s - %(asctime)s]: %(message)s",
        filename=args.log_file
    )

    if not args.verbose or args.verbose == 0:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

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
