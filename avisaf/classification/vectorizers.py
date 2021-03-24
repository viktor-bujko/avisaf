#!/usr/bin/env python3

import numpy as np
import spacy
import logging
import concurrent.futures
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import utils
from gensim.models import Doc2Vec, Word2Vec, TfidfModel
from gensim.models.doc2vec import LabeledSentence
from gensim.corpora import Dictionary

logging.basicConfig(
    level=logging.DEBUG,
    format=f'[%(levelname)s - %(asctime)s]: %(message)s'
)


class AsrsReportVectorizer:

    def build_feature_vectors(self, texts: type(np.ndarray), target_labels_shape: int, train: bool = False) -> np.ndarray:
        """
        :param train:
        :type train: bool
        :param texts:
        :type texts: np.ndarray
        :param target_labels_shape:
        :type target_labels_shape: int
        :return: Feature vectors for given texts.
        :rtype: np.ndarray
        """
        pass

    def get_params(self):
        pass


class TfIdfAsrsReportVectorizer(AsrsReportVectorizer):

    def __init__(self):
        self.transformer_name = 'tfidf'
        self._transformer = TfidfVectorizer(
            stop_words='english',
            lowercase=False,
            ngram_range=(1, 3)
        )

    def build_feature_vectors(self, texts: type(np.ndarray), target_labels_shape: int, train: bool = False):

        if texts.shape[0] != target_labels_shape:
            msg = 'The number of training examples is not equal to the the number of labels.'
            logging.error(msg)
            logging.error(f'Texts.shape: {texts.shape[0]} vs labels.shape: {target_labels_shape}')
            raise ValueError(msg)

        """logging.debug("Starting vectorization")

        batch = int(texts.shape[0] / 5)

        processed_texts_1 = []
        processed_texts_2 = []
        processed_texts_3 = []
        processed_texts_4 = []
        processed_texts_5 = []

        with concurrent.futures.ThreadPoolExecutor() as exec:
            future_1 = exec.submit(self.lemmatize_and_remove_stops, texts[0: batch], 0)
            future_2 = exec.submit(self.lemmatize_and_remove_stops, texts[batch: batch * 2], batch)
            future_3 = exec.submit(self.lemmatize_and_remove_stops, texts[batch * 2: batch * 3], (batch * 2))
            future_4 = exec.submit(self.lemmatize_and_remove_stops, texts[batch * 3: batch * 4], (batch * 3))
            future_5 = exec.submit(self.lemmatize_and_remove_stops, texts[batch * 4:], (batch * 4))

            processed_texts_1 = np.char.join(' ', future_1.result())
            processed_texts_2 = np.char.join(' ', future_2.result())
            processed_texts_3 = np.char.join(' ', future_3.result())
            processed_texts_4 = np.char.join(' ', future_4.result())
            processed_texts_5 = np.char.join(' ', future_5.result())

        processed_texts = np.concatenate(
            (
                np.concatenate((processed_texts_1, processed_texts_2, processed_texts_3)),
                np.concatenate((processed_texts_4, processed_texts_5))
            )
        )

        logging.debug("Ended pre-processing")"""

        if train:
            # TODO: Try different types of feature extractors / word embeddings
            texts_vectors = self._transformer.fit_transform(texts)  # TODO: .toarray() call -> create a matrix from csr_matrix
        else:
            texts_vectors = self._transformer.transform(texts)

        logging.debug("Ended vectorization")
        return texts_vectors

    def lemmatize_and_remove_stops(self, texts, start_index: int = 0):
        nlp = spacy.load('en_core_web_md')
        processed_texts = []
        processed = start_index
        for text in texts:
            processed += 1
            if processed % 250 == 0:
                logging.debug(f'Processing text {processed}')
            doc = nlp(str(text))
            processed_text = []
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    processed_text.append(token.lemma_)
            processed_texts.append(processed_text)

        return processed_texts

    def get_params(self):
        return {
            "encoding": self._transformer.encoding,
            "decode_error": self._transformer.decode_error,
            "stop_words": self._transformer.stop_words,
            "lowercase": self._transformer.lowercase,
            "ngram_range": self._transformer.ngram_range,
            "max_features": self._transformer.max_features,
            "analyzer": self._transformer.analyzer,
            "tokenizer": self._transformer.tokenizer,
            "preprocessor": self._transformer.preprocessor,
            "use_idf": self._transformer.use_idf,
            "norm": self._transformer.norm,
        }


class Word2VecAsrsReportVectorizer(AsrsReportVectorizer):

    def __init__(self):
        pass

    def build_feature_vectors(self, texts: type(np.ndarray), target_labels_shape: int, train: bool = False):
        nlp = spacy.load('en_core_web_md')

        for text in texts:
            doc = nlp(str(text))
            for token in doc:
                if token.is_stop:
                    print(f'Lemma: {token.lemma_}')
            input()

    def get_params(self):
        return {}
