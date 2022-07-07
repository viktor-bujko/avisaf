#!/usr/bin/env python3
import sys
from pathlib import Path

import spacy
import logging
import numpy as np
from spacy import displacy as displacy
import avisaf.util.data_extractor as de

logger = logging.getLogger("avisaf_logger")


def get_options():
    ents_colors_dict = de.get_entities()

    options = {
        "ents": list(ents_colors_dict.keys()),
        "colors": dict(
            zip(
                ents_colors_dict.keys(),
                map(lambda color_list: color_list[0], ents_colors_dict.values()),
            )
        ),
    }

    return options


def print_to_console(document):
    ents_colors_dict = de.get_entities()

    print()  # separate output text

    def print_highlighted_entity(tkn):
        if not tkn.ent_type_:
            print(f"{tkn.text} ", end="")
        else:
            color = ents_colors_dict.get(tkn.ent_type_)[1]
            print(f"\x1b[{color}m[{tkn.text}: {tkn.ent_type_}]\033[0m ", end="")

    for token in document:
        print_highlighted_entity(token)
    print()  # to end the text


def render_to_html(document, html_result_file):
    options = get_options()

    if html_result_file and not html_result_file.endswith(".html"):
        html_result_file += ".html"
    result_file_path = Path(html_result_file)
    result_file_path.touch(exist_ok=True)
    with result_file_path.open(mode="w") as file:
        html = displacy.render(document, style="ent", options=options)
        file.write(html)


def extract_text_to_process(text_path) -> str:
    if text_path is None:
        sample_text_path = str(Path("config", "sample_text.txt"))
        txt_ext = de.TextFileExtractor([sample_text_path])
    else:
        # extract the text
        text_path = Path(text_path)
        if not text_path.exists():
            logger.error(f"Requested file \"{str(text_path)}\" does not exist.")

        # file exists
        suffix = text_path.suffix
        suffix_extractor = {
            ".txt": de.TextFileExtractor,
            ".csv": de.CsvAsrsDataExtractor,
            ".json": de.JsonDataExtractor,
        }
        extractor = suffix_extractor.get(suffix)
        if not extractor:
            logger.error(f"Please make sure given file uses one of the following formats: {list(suffix_extractor.keys())}")
            return None

        txt_ext = extractor([text_path])

    texts = txt_ext.extract_data(["Report 1_Narrative"]).get("Report 1_Narrative")
    texts = np.array(texts)  # unification of lists format
    if texts.shape[0] == 0:
        logger.error("An error occured during report narrative extraction.")
        return None
    else:
        return texts.tolist()


def process_ner(
    model="en_core_web_md",
    text_path=None,
    text: str = None,
    cli_result: bool = False,
    visualize: bool = False,
    html_result_file: str = None,
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
    :param text_path: String representing a path to the file with the
        text to have named entities extracted.
    :type text: str
    :param text: String representing a narrative text, which will have its named
        entities extracted.
    :type cli_result: bool
    :param cli_result: A flag which will cause the result to be printed to the
        stdout., defaults to False
    :type visualize: bool
    :param visualize: A flag which will use the spaCy visualizer in order to
        render the result and show it in the browser., defaults to False
    :type html_result_file: str
    :param html_result_file: The file path to the file where the result rendered
        by spaCy visualizer tool will be saved. The file will be created if it
        does not exist yet., defaults to None
    :type port: int
    :param port: The number of the port to be used for displaCy rendered
        visualization. Argument only used when visualize is true or html_result_file
        is defined.
    """

    if text is None:
        texts = extract_text_to_process(text_path)
    else:
        texts = [text]

    if texts is None:
        logger.error("A suitable report narrative could not be extracted.")
        return

    # create new nlp object
    if model.startswith("en_core_web"):
        logger.info("Using a default english language model!")

    # loading either the pre-trained spaCy model or a model in given directory
    nlp = spacy.load(model)

    if not nlp.has_pipe("ner"):
        logger.error(f"The model '{model}' is not available or does not contain required components.")
        return

    # create doc object nlp(text)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    for document in nlp.pipe(texts, disable=other_pipes):

        # identify entities
        if cli_result or len(texts) > 1:
            print_to_console(document)
            continue

        if html_result_file is None and not visualize:
            logger.warning("Named-entity recognition output method is not defined. Please use --help to get info about possible output visualizations.")
            break

        if html_result_file is None:
            displacy.serve(
                document,
                style="ent",
                options=get_options(),
                port=port,
                host="localhost"
            )
        else:
            render_to_html(document, html_result_file)
