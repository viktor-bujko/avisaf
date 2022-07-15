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
        "--config",
        type=str,
        default=str(Path("config", "spacy_ner.cfg")),
        help="Path to the spaCy configuration file (see https://spacy.io/api/data-formats#config)."
    )
    parser.add_argument(
        "-d",
        "--data",
        metavar="DATA PATH",
        nargs="+",
        help="Path to the files with annotated JSON training data.",
        default=[str(Path("data_files", "ner", "train_data", "annotated_data_01.json"))]
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="PATH/MODEL",
        help="File path to an existing spaCy model or existing spaCy model name to be trained.",
        default=None,
    )
    parser.add_argument(
        "-n", "--name",
        metavar="MODEL NAME",
        help="Name of the new model. Will be generated if not provided.",
        default=None
    )

    return parser


def add_ner_tester_parser(subparsers):
    """Method responsible for parsing ner test subcommand and its arguments"""

    parser = subparsers.add_parser(
        "ner_process",
        help="Test a selected model.",
        description="Command used for testing the entity recognition on given text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.set_defaults(dest="process_ner")
    parser.add_argument(
        "-m",
        "--model",
        metavar="PATH/MODEL",
        default=str(Path("models", "ner", "ner_avisaf_model")),
        help="File path to an existing spaCy model or existing spaCy model name for NER.",
    )
    text_group = parser.add_mutually_exclusive_group(required=False)
    text_group.add_argument(
        "-t",
        "--text",
        default=None,
        type=str,
        help="Report narrative which will have entities extracted. If None, sample text is used.",
    )
    text_group.add_argument(
        "--text_path",
        help="String representing a path to .txt file containing reports to have their classes predicted.",
        default=str(Path("config", "sample_text.txt")),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port number to be used for rendering (ignored if -r nor -s are used).",
    )

    visualization_group = parser.add_mutually_exclusive_group(required=True)
    visualization_group.add_argument(
        "-p", "--print", action="store_true", help="Print the result on the screen."
    )
    visualization_group.add_argument(
        "-r",
        "--render",
        action="store_true",
        help="A flag to indicate whether a visualization tool should be started.",
    )
    visualization_group.add_argument(
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
        metavar="PATH/MODEL",
        default=str(Path("models", "ner", "ner_avisaf_model")),
        help="File path to an existing spaCy model or existing spaCy model name for NER.",
    )
    parser.add_argument(
        "-t",
        "--texts",
        metavar="PATH",
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
        "-d", "--data",
        type=str,
        help="Training data source file path.",
        default=None,
        required=True
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
        "--texts",
        help="The path to the file containing texts to be annotated. (Supports .csv files only).",
        default=None,
        required=True
    )
    parser.add_argument(
        "--lines",
        type=int,
        help="The number of texts to be annotated (1 text = 1 line). Default value means all texts",
        default=-1,
    )
    parser.add_argument(
        "-l",
        "--labels",
        help="Path to the file containing entity labels used for annotation.",
        default=str(Path("config", "entities_labels.json")),
    )
    parser.add_argument(
        "-s",
        "--start_index",
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
        "classifier_train",
        help="Train an ASRS reports classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.set_defaults(dest="classifier_train")
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Strings representing the paths to training data texts",
        default=[],
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
        choices={"knn", "svm", "mlp"},
    )
    parser.add_argument(
        "--params_overrides",
        nargs="+",
        help="Override default algorithm hyper-parameters. Use \"param_key=param_value\" from scikit documentation (>= 1 \"=\" character required per pair)",
        default=[]
    )
    parser.add_argument(
        "--normalize",
        "-n",
        choices={"undersample", "oversample"},
        default=None,
        help="Normalize the distribution of classes in training data",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=[],
        nargs="+",
        help="Trained model(s) to use (at least one is required)",
    )
    parser.add_argument(
        "-v",
        "--vectorizer",
        choices={"tfidf", "spacyw2v", "googlew2v", "d2v"},
        default=None
    )
    parser.add_argument(
        "--set_default_class",
        action="store_true",
        help="Sets default text target class as \"Other\". Ignored if --filter list is not defined.",
        default=False
    )
    return parser


def add_classification_processing_parser(subparsers):

    parser = subparsers.add_parser(
        "classifier_process",
        help="Apply NER model predictions for classes assignment to reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.set_defaults(dest="classifier_process")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text_path",
        help="String representing a path to .txt file containing reports to have their classes predicted.",
        default=None,
    )
    group.add_argument(
        "-t", "--text",
        help="Report narrative text used for classification.",
        default=None
    )
    parser.add_argument(
        "-m",
        "--model",
        default=str(Path("models", "classifiers", "avisaf_classifier_model")),
        help="Path of a model used for text classification."
    )

    return parser


def add_classification_evaluation_parser(subparsers):
    parser = subparsers.add_parser(
        "classifier_eval",
        help="Evaluate an ASRS reports classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.set_defaults(dest="classifier_eval")
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Strings representing the paths to training data texts",
        default=[],
    )
    parser.add_argument(
        "-m",
        "--model",
        default=str(Path("models", "classifiers", "avisaf_classifier_model")),
        help="Path of a trained model to be tested."
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        default=False,
        help="Compare given prediction model with baseline dummy and random predictors."
    )
    parser.add_argument(
        "--show_curves",
        action="store_true",
        default=False,
        help="Show ROC and Precision-Recall curves for evaluated model."
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
    main_parser.add_argument(
        "--log_file",
        default=None,
        help="A string which defines the log file path."
    )
    subparser = main_parser.add_subparsers(help="Possible operations to perform.")

    for parser_func in [
        add_ner_trainer_parser,
        add_ner_tester_parser,
        add_ner_evaluator_parser,
        add_auto_annotator_parser,
        add_manual_annotator_parser,
        add_classification_train_parser,
        add_classification_processing_parser,
        add_classification_evaluation_parser
    ]:
        parsers_list.append(parser_func(subparser))

    available_parsers = dict(map(lambda parser: (parser.prog.split()[1], parser), parsers_list))  # parser.prog is a `avisaf operation` string

    return main_parser, available_parsers
