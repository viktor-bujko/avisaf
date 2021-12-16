#!/usr/bin/env python3

import argparse
from pathlib import Path


def add_ner_trainer_parser(subparsers):
    """Method responsible for parsing ner train subcommand and its arguments"""

    parser = subparsers.add_parser(
        "ner_train",
        help="Train a new NLP NER model.",
        description="Command for training new/updating entities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.set_defaults(dest="train_ner")
    parser.add_argument(
        "-d",
        "--data",
        metavar="PATH",
        nargs="+",
        help="File path(s) to the file with annotated training data.",
        default=[Path("data_files", "ner", "train_data", "annotated_data_01.json")],
        required=True,
    )
    parser.add_argument(
        "-i",
        "--iterations",
        metavar="INT",
        type=int,
        default=20,
        help="The number of iterations to perform for entity training.",
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="PATH/NAME",
        help="File path to an existing spaCy model or existing spaCy model name to be trained.",
        default=None,
    )
    parser.add_argument(
        "-n", "--name",
        metavar="STRING",
        help="Name of the new model.", default=None
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size."
    )

    return parser


def add_ner_tester_parser(subparsers):
    """Method responsible for parsing ner test subcommand and its arguments"""

    parser = subparsers.add_parser(
        "ner_test",
        help="Test a selected model.",
        description="Command used for testing the entity recognition on given text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.set_defaults(dest="test_ner")
    parser.add_argument(
        "-m",
        "--model",
        metavar="PATH/MODEL",
        default="en_core_web_md",
        required=True,
        help="File path to an existing spaCy model or existing spaCy model name for NER.",
    )
    parser.add_argument(
        "-t",
        "--text",
        default=None,
        help="File path to the text which will have entities extracted. If None, sample text is used.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port number to be used for rendering (ignored if -r nor -s are used).",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p", "--print", action="store_true", help="Print the result on the screen."
    )
    group.add_argument(
        "-r",
        "--render",
        action="store_true",
        help="A flag to indicate whether a visualization tool should be started.",
    )
    group.add_argument(
        "-s",
        "--save",
        metavar="PATH",
        default=None,
        help="Save rendered html result into the file (will be created if does not exist).",
    )

    return parser


def add_ner_evaluator_parser(subparsers):
    parser = subparsers.add_parser(
        "ner_eval",
        help="Evaluate a selected model.",
        description="Command used for evaluation performance of the entity recognition model on given set of texts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.set_defaults(dest="eval_ner")
    parser.add_argument(
        "-m",
        "--model",
        metavar="PATH TO MODEL",
        default=None,
        required=True,
        help="File path to an existing spaCy model or existing spaCy model name for NER.",
    )
    parser.add_argument(
        "-t",
        "--texts",
        metavar="PATH TO JSON TEST FILES",
        default=None,
        required=True,
        help="File path to the texts JSON file which will be considered as gold dataset.",
    )

    return parser


def add_auto_annotator_parser(subparsers):
    """Method responsible for parsing auto-annotation tool and its arguments."""

    parser = subparsers.add_parser(
        "annotate_auto",
        help="Automatic annotation tool for new training dataset creation.",
        description="Automatic annotation tool for new training dataset creation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.set_defaults(dest="annotate_auto")
    parser.add_argument("keys_file", help="Path to file with words to be matched.")
    parser.add_argument("label", type=str, help="The text of the label of an entity.")
    parser.add_argument(
        "-d", "--data", type=str, help="Training data source file path.", default=None
    )
    parser.add_argument(
        "-e",
        "--extract",
        action="store_true",
        help="Flag indicating that text extraction should take place. Otherwise, JSON file containing list of ('report string', entities dictionary) tuples is required.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="File path to an existing spaCy model or existing spaCy model name.",
        default="en_core_web_md",
    )
    parser.add_argument(
        "-p",
        action="store_true",
        help="Flag indicating that spaCy's PhraseMatcher object should be used.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Flag indicating that the result should be saved. Requires the -d/--data argument.",
    )
    parser.add_argument(
        "--save_to",
        default=None,
        help="String representing the path where the result should be saved. If None (default), -d/--data argument value will be used. Ignored if -s/--save is not used."
    )

    return parser


def add_manual_annotator_parser(subparsers):
    """Method responsible for parsing manual-annotation tool and its arguments."""

    parser = subparsers.add_parser(
        "annotate_man",
        help="Manual annotation tool for new training dataset creation.",
        description="Manual annotation tool for new training dataset creation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.set_defaults(dest="annotate_man")
    parser.add_argument(
        "texts_file",
        help="""The path to the file containing texts to be annotated. (Supports .csv/.json files). If None, then a user can write own sentences and annotate them.""",
        default=None,
    )
    parser.add_argument(
        "lines", type=int, help="The number of texts to be annotated (1 text = 1 line)."
    )
    parser.add_argument(
        "-l",
        "--labels",
        help="Path to the file containing entity labels used for annotation.",
        default=Path("entities_labels.json"),
    )
    parser.add_argument(
        "-s",
        "--start-index",
        type=int,
        help="The index of the first text to be annotated.",
        default=0,
    )
    parser.add_argument(
        "--not_save",
        action="store_false",
        help="Flag indicating whether the result of the annotation should NOT be saved.",
    )

    return parser


def add_classification_train_parser(subparsers):
    """Method responsible for parsing classification training command and its arguments."""

    parser = subparsers.add_parser(
        "classifier",
        help="Train an ASRS reports classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.set_defaults(dest="classifier")
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Strings representing the paths to training data texts",
        default=[],
    )
    parser.add_argument(
        "--mode",
        choices={"train", "dev", "test"},
        default="test",
        help="Choose classifier operating mode (default test)",
    )
    parser.add_argument(
        "-l",
        "--label",
        help="The label of the column to be extracted from the documents (in format FirstLineLabel_SecondLineLabel)",
        default=None,
    )
    parser.add_argument(
        "-f",
        "--filter",
        nargs="*",
        help="Subset of the values present in the column given by the label",
        default=None,
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        default="mlp",
        help="The algorithm used for classification training.",
        choices={"knn", "svm", "mlp", "forest", "gauss", "mnb", "regression"},
    )
    # TODO: Enable --oversample / --undersample arguments instead
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the distribution of classes in training data",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show AUC for each of selected models"
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        nargs="+",
        help="Trained model(s) to use (at least one is required)",
    )

    return parser


def parse_args() -> tuple:

    parsers_list = []

    main_parser = argparse.ArgumentParser(
        description="A tool for aviation safety reports entity recognition and text classification.",
        prog="avisaf",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    main_parser.set_defaults(verbose=0)
    main_parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Verbosity level setting. More -v occurrences increase the number of printed messages."
    )
    subparser = main_parser.add_subparsers(help="Possible operations to perform.")

    for parser_func in [
        add_ner_trainer_parser,
        add_ner_tester_parser,
        add_ner_evaluator_parser,
        add_auto_annotator_parser,
        add_manual_annotator_parser,
        add_classification_train_parser
    ]:
        parsers_list.append(
            parser_func(subparser)
        )

    available_parsers = dict(
        map(lambda parser: (parser.prog.split()[1], parser), parsers_list)
    )  # parser.prog is a `avisaf operation` string

    return main_parser, available_parsers
