#!/usr/bin/env python3

import spacy
import logging
from spacy.training.example import Example
from util.data_extractor import JsonDataExtractor

logger = logging.getLogger("avisaf_logger")


@spacy.registry.readers("train_data_stream")
def stream_data(data_source: list):
    logger.debug(f"Training data source files list: {data_source}")
    extractor = JsonDataExtractor(data_source)
    # data_to_stream =

    def stream(nlp):
        for annotated_texts in extractor.get_ner_training_data():
            for text, ents in annotated_texts:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, ents)
                yield example

    return stream