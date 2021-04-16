#!/usr/bin/env python3

import numpy as np
import spacy
import logging
import sys
import re
import concurrent.futures
from pathlib import Path
from spacy.language import Doc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from gensim import utils
import gensim.downloader as dnld
from gensim.test.utils import datapath
from gensim.models import Doc2Vec, TfidfModel, Word2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from gensim.models.fasttext import FastText
from gensim.corpora import Dictionary


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
            lowercase=True,
            ngram_range=(1, 3),
            analyzer='word',
            min_df=0.02,
            max_features=10,
            max_df=0.5
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

    @property
    def transformer(self):
        return self._transformer


class Doc2VecAsrsReportVectorizer(AsrsReportVectorizer):

    def __init__(self):
        pass

    def build_feature_vectors(self, texts: type(np.ndarray), target_labels_shape: int, train: bool = False):

        try:
            model = Doc2Vec.load('doc2vec.model')
        except FileNotFoundError:
            model = None

        tagged_docs = []
        for idx, text in enumerate(texts):
            tokens = utils.simple_preprocess(text)
            tagged_docs.append(TaggedDocument(tokens, [idx]))

        if model is None:
            if model is None:
                model = Doc2Vec(vector_size=200, epochs=40, min_count=2)
                model.build_vocab(documents=tagged_docs)

            model.train(
                documents=tagged_docs,
                total_examples=model.corpus_count,
                epochs=model.epochs
            )
            model.save('doc2vec.model')

        doc2veced = np.ndarray((texts.shape[0], model.vector_size))
        if len(tagged_docs) != texts.shape[0]:
            raise ValueError('Incorrect dimensions of tagged_docs and texts on input')

        doc2veced_count = 0
        for idx, tagged_doc in enumerate(tagged_docs):
            doc2veced_count += 1
            if doc2veced_count % 1000 == 0:
                logging.debug(doc2veced_count)
            doc2veced[idx] = model.infer_vector(tagged_doc.words)

        return doc2veced

    def get_params(self):
        return {}


class Word2VecAsrsReportVectorizer(AsrsReportVectorizer):

    def __init__(self, vectors=None):
        self._nlp = spacy.load('en_core_web_md')
        self._vectors = vectors

    @staticmethod
    def preprocess(texts):

        preprocessed = []
        for text in texts:
            text = text.lower()
            text = re.sub(r'([0-9]{1,2});([0-9]{1,3})', r'\1,\2', text)     # ; separated numbers - usually altitude
            text = re.sub(r'fl[0-9]{2,3}', 'flight level', text)            # flight level representation
            text = re.sub(r'runway|rwy [0-9]{1,2}[rcl]?', r'runway', text)  # runway identifiers
            text = re.sub(r'([a-z]*)[?!\-.]([a-z]*)', r'\1 \2', text)       # "word[?!/-.]word" -> "word word"
            text = re.sub(r'(z){3,}[0-9]*', r'airport', text)               # anonymized "zzz" airports
            text = re.sub(r'tx?wys?', 'taxiway', text)
            text = re.sub(r'twrs?[^a-z]', 'tower', text)
            text = re.sub('tcas', 'traffic collision avoidance system', text)
            text = re.sub(r'([a-z0-9]+\.){2,}[a-z0-9]*', '', text)          # removing words with several dots
            text = re.sub(r'(air)?spds?', 'speed', text)
            text = re.sub(r'qnh', 'pressure', text)
            text = re.sub(r'lndgs?', 'landing', text)
            preprocessed.append(text)

        return preprocessed

    def build_feature_vectors(self, texts: list, target_labels_shape: int, train: bool = False) -> np.ndarray:
        logging.debug('Started vectorization')

        texts = self.preprocess(texts)

        doc_vectors = []
        for doc_vector_batch in self._generate_vectors(texts, self._vectors, 256):
            doc_vectors.append(doc_vector_batch)
            print("===========================")

        result = np.array(np.concatenate(doc_vectors, axis=0))
        logging.debug(f'Vectorized {result.shape[0]} texts')

        return result

    def _generate_vectors(self, texts, vectors, batch: int = 50):
        oov = set()
        doc_vectors = []
        for doc in self._nlp.pipe(texts, disable=['ner'], batch_size=batch):
            lemmas = []
            for token in doc:
                # if token.conjuncts:
                # print(f'"{token.text}" conjuncts: {[conj.text for conj in token.conjuncts]}')
                if token.is_punct:
                    # ignoring the punctuation
                    continue
                if token.pos_ == 'PRON' and token.text in vectors:
                    # taking "we", "he" etc instead of "-PRON-"
                    lemmas.append(token.text)
                    continue
                if token.is_stop:
                    # ignoring the stop words, but **after** taking the pronouns
                    continue
                if token.like_num:
                    # replacing all numbers by common tag
                    lemmas.append("number")
                    continue
                if token.lemma_ not in vectors:
                    # trying to identify some common mistakes
                    words = re.sub(r'([a-z]*)[?!\-.]([a-z]*)', r'\1 \2', token.text).split()  # [?!\-.] separated words replaced by space
                    for word in words:
                        if word not in vectors:
                            continue
                        lemmas.append(word)

                    # ignoring word which don't have vector representation
                    logging.warning(f'Word "{token.text}" with lemma "{token.lemma_} not in vocabulary"')
                    oov.add(token.text)
                    continue

                lemmas.append(token.lemma_)

            doc_vector = self._get_doc_vector(lemmas)  #
            doc_vectors.append(doc_vector)

        yield np.array(doc_vectors)

        with open('out_of_vocab_20210416.txt', 'w') as oov_file:
            logging.debug("Saving unused words")
            print(*sorted(list(oov)), sep='\n', file=oov_file)

    def _get_doc_vector(self, lemmas):
        pass


class SpaCyWord2VecAsrsReportVectorizer(Word2VecAsrsReportVectorizer):
    def __init__(self):
        logging.debug("Loading spacy model")
        self._nlp = spacy.load('en_core_web_md')
        super().__init__(self._nlp.vocab)

    def _get_doc_vector(self, lemmas):
        return Doc(self._nlp.vocab, words=lemmas).vector * (len(lemmas) / 500)

    def get_params(self):
        return {
            "name": "SpaCyWord2Vec"
        }


class GoogleNewsWord2VecAsrsReportVectorizer(Word2VecAsrsReportVectorizer):
    def __init__(self):
        logging.debug(Path().absolute())
        self._model_path = Path('gensim-data', 'GoogleNews-vectors-negative300.bin')
        if not self._model_path.exists():
            logging.warning("Pre-trained GoogleNews model used for vectorization has not been found.")
            if input("Do you want to download and unzip the model (1.5 Gb zipped size)? (y/N) ").lower() == 'y':
                logging.debug('(down)LOADING')
                self._model_path = dnld.load('word2vec-google-news-300', return_path=True)
                print(f'MODEL PATH: {self._model_path}')
            else:
                sys.exit(1)

        # TODO: Download and unzip the model

        print("Loading large GoogleNews model. May take a while.")
        self._model = KeyedVectors.load_word2vec_format(self._model_path, binary=True)
        super().__init__(self._model.wv)

    def _get_doc_vector(self, lemmas):
        return np.mean(self._model[lemmas] * (len(lemmas) / 500), axis=0)

    def get_params(self):
        return {
            "name": "GoogleNewsWord2Vec",
            "model_path": str(self._model_path)
        }


if __name__ == '__main__':
    x = Word2VecAsrsReportVectorizer()
    # x = GoogleNewsWord2VecAsrsReportVectorizer()
    result = x.build_feature_vectors([
        "on approach; captain (pilot flying) called for flaps 8; and i positioned flap lever to 8 position. an amber eicas message 'flaps fail' annunciated with a master caution with the flaps failed at the 0 degree position.",
        'during climb to 17;000 feet the first officer noticed subtle propeller fluctuations.',
        'i was conducting ojt with a developmental that has 1 r-side and all d-sides.',
        'ocean west and offshore west/central were combined.',
        'during climb to 17;000 ft the first officer noticed subtle propeller fluctuations.',
    ], 4)

    print(result)
