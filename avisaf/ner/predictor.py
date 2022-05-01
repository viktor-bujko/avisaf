#!/usr/bin/env python3
from pathlib import Path

import spacy
import logging
from spacy import displacy as displacy
from util.data_extractor import get_entities

logger = logging.getLogger("avisaf_logger")


def test_ner(
    model="en_core_web_md",
    text_path=None,
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
    :param text_path: String representing either a path to the file with the
        text to be inspected or the text itself., defaults to None
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

    if text_path is None:
        # use sample text
        with Path("config", "sample_text.txt").open("r") as sample_text_file:
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
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    document = nlp(text, disable=other_pipes)

    ents_colors_dict = get_entities()
    # identify entities
    if cli_result:
        print()  # separate output text

        def print_highlighted_entity(tkn):
            # TODO: použi tkn.ent_iob namiesto ent_type_ pre rozpoznanie viactokenových entít
            if not tkn.ent_type_:
                print(f"{tkn.text} ", end="")
            else:
                color = ents_colors_dict.get(tkn.ent_type_)[1]
                print(f"\x1b[{color}m[{tkn.text}: {tkn.ent_type_}]\033[0m ", end="")

        for token in document:
            # TODO: použi nejaký decorator na poskytovanie rôznych vypisovaní
            print_highlighted_entity(token)
        print()  # to end the text
        return

    if html_result_file is None and not visualize:
        logger.warning("Named-entity recognition output method is not defined. Please use --help to get info about possible output visualizations.")
        return

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

    if html_result_file and not html_result_file.endswith(".html"):
        html_result_file += ".html"
    result_file_path = Path(html_result_file)
    result_file_path.touch(exist_ok=True)
    with result_file_path.open(mode="w") as file:
        html = displacy.render(document, style="ent", options=options)
        file.write(html)