#!/usr/bin/env python3

import numpy as np
import spacy
import logging
import sys
import re
from pathlib import Path
from spacy.language import Doc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import utils
import gensim.downloader as dwnldr
from gensim.models import Doc2Vec, Word2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from gensim.models.fasttext import FastText


def show_vector_space_3d(vectors, targets):

    assert vectors.shape[0] == targets.shape[0]

    pca = PCA(n_components=3)
    components = pca.fit_transform(vectors)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.patch.set_alpha(0.5)
    ax.grid()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    unique_targets = np.unique(targets)
    ax.legend(unique_targets)
    colors = ['r', 'g', 'b']

    for target, color in zip(unique_targets, colors):
        ax.scatter(components[targets == target][:, 0],
                   components[targets == target][:, 1],
                   components[targets == target][:, 2],
                   c=color,
                   s=50
                   )

    plt.xlabel('PC 1', size=15)
    plt.ylabel('PC 2', size=15)
    plt.title('Word embedding space', size=20)

    plt.show()
    show_vector_space_2d(vectors, targets)


def show_vector_space_2d(vectors, targets):

    assert vectors.shape[0] == targets.shape[0]

    pca = PCA(n_components=2)
    components = pca.fit_transform(vectors)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    ax.patch.set_alpha(0.5)
    ax.grid()
    plt.xlim(-4, 20)
    plt.ylim(-7, 20)

    unique_targets = np.unique(targets)
    ax.legend(unique_targets)
    colors = ['r', 'g', 'b']

    for target, color in zip(unique_targets, colors):
        ax.scatter(components[targets == target][:, 0],
                   components[targets == target][:, 1],
                   c=color,
                   s=50
                   )

    plt.xlabel('PC 1', size=15)
    plt.ylabel('PC 2', size=15)
    plt.title('Word embedding space', size=20)

    plt.show()


class AsrsReportVectorizer:

    def build_feature_vectors(self, texts: np.ndarray, target_labels_shape: int, train: bool = False) -> np.ndarray:
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

    @staticmethod
    def preprocess(texts):
        logging.debug("Started preprocessing")
        preprocessed = []
        for text in texts:
            text = re.sub(r' [A-Z]{5} ', ' waypoint ', text)    # replacing uppercase 5-letter words - most probably waypoints
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

        logging.debug("Ended preprocessing")

        return preprocessed

    def get_params(self):
        pass


class TfIdfAsrsReportVectorizer(AsrsReportVectorizer):

    def __init__(self):
        self.transformer_name = 'tfidf'
        self._transformer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 4),
            analyzer='word',
            max_features=50_000,
            # max_df=0.5
        )
        self._pipeline = Pipeline([
            ('reductor', TruncatedSVD(n_components=300)),
            ('scaler', StandardScaler())
        ])

    def build_feature_vectors(self, texts: type(np.ndarray), target_labels: type(np.ndarray), train: bool = False):

        if texts.shape[0] != target_labels.shape[0]:
            msg = 'The number of training examples is not equal to the the number of labels.'
            logging.error(msg)
            logging.error(f'Texts.shape: {texts.shape[0]} vs labels.shape: {target_labels}')
            raise ValueError(msg)

        texts = self.preprocess(texts)

        import lzma
        import pickle

        if train:
            texts_vectors = self._transformer.fit_transform(texts)
            texts_vectors = self._pipeline.fit_transform(texts_vectors)
            with lzma.open('pipeline.model', 'wb') as pipe:
                pickle.dump((self._transformer, self._pipeline), pipe)
            # texts_vectors = self._transformer.fit_transform(texts)
        else:
            with lzma.open('pipeline.model', 'rb') as pipe:
                self._transformer, self._pipeline = pickle.load(pipe)
            texts_vectors = self._transformer.transform(texts)
            texts_vectors = self._pipeline.transform(texts_vectors)
            # texts_vectors = self._transformer.transform(texts)

        logging.debug("Ended vectorization")
        return texts_vectors

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

    def build_feature_vectors(self, texts: type(np.ndarray), target_labels: type(np.ndarray), train: bool = False):

        try:
            model = Doc2Vec.load('doc2vec.model')
        except FileNotFoundError:
            model = None

        tagged_docs = []
        texts = np.array(self.preprocess(texts))

        assert texts.shape[0] == target_labels.shape[0]

        for text, label in zip(enumerate(texts), target_labels):
            idx, text = text
            tokens = utils.simple_preprocess(text)
            tagged_docs.append(TaggedDocument(words=tokens, tags=[label]))

        if model is None:
            model = Doc2Vec(
                vector_size=300,
                epochs=40,
                min_count=5,
                dm=0,
                dbow_words=1,
                negative=5,
                workers=3,
                window=4,
                alpha=0.001,
                min_alpha=0.0001
            )
            logging.debug(model.estimate_memory())
            model.build_vocab(documents=tagged_docs)

            model.train(
                documents=tagged_docs,
                total_examples=model.corpus_count,
                epochs=model.epochs
            )
            model.save('doc2vec.model')

        doc2veced = np.zeros(shape=(texts.shape[0], model.vector_size))
        if len(tagged_docs) != texts.shape[0]:
            raise ValueError('Incorrect dimensions of tagged_docs and texts on input')

        doc2veced_count = 0
        for idx, tagged_doc in enumerate(tagged_docs):
            doc2veced_count += 1
            if doc2veced_count % 1000 == 0:
                logging.debug(doc2veced_count)
            doc2veced[idx] = model.infer_vector(tagged_doc.words, epochs=6)
        return doc2veced

    def get_params(self):
        return {}


class Word2VecAsrsReportVectorizer(AsrsReportVectorizer):

    def __init__(self, vectors=None):
        self._nlp = spacy.load('en_core_web_md')
        self._vectors = vectors

    def build_feature_vectors(self, texts: np.ndarray, target_labels_shape: int, train: bool = False) -> np.ndarray:
        logging.debug('Started vectorization')

        texts = self.preprocess(texts)

        doc_vectors = []
        for doc_vector_batch in self._generate_vectors(texts, self._vectors, 256):
            doc_vectors.append(doc_vector_batch)
            print("===========================")

        result = np.concatenate(doc_vectors, axis=0)
        logging.debug(f'Vectorized {result.shape[0]} texts')
        """

        import lzma
        import pickle

        with lzma.open('vectorsdev.vec', 'rb') as f:
            # logging.debug('Saving vectors')
            # pickle.dump(result, f)
            result = pickle.load(f)"""

        return result

    def _generate_vectors(self, texts, vectors, batch: int = 100):
        oov = set()
        doc_vectors = []
        for doc in self._nlp.pipe(texts, disable=['ner'], batch_size=batch):
            lemmas = []
            for token in doc:
                if token.pos_ == 'PRON' and token.text in vectors:
                    # taking "we", "he" etc instead of "-PRON-"
                    lemmas.append(token.text)
                    continue
                if token.is_punct or token.is_stop or str(token.text).isspace() or not str(token.text):
                    # ignoring the punctuation and empty/whitespace tokens
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
                    if token.text not in oov:
                        logging.warning(f'Word "{token.text}" with lemma "{token.lemma_}" not in vocabulary')
                        oov.add(token.text)
                    continue

                lemmas.append(token.lemma_)

            doc_vector = self._get_doc_vector(lemmas)  #
            doc_vectors.append(doc_vector)

        yield np.array(doc_vectors)

        # with open('out_of_vocab_20210421.txt', 'w') as oov_file:
        #    logging.debug("Saving unused words")
        #    print(*sorted(list(oov)), sep='\n', file=oov_file)

    def _get_doc_vector(self, lemmas):
        pass


class SpaCyWord2VecAsrsReportVectorizer(Word2VecAsrsReportVectorizer):
    def __init__(self):
        logging.debug("Loading spacy model")
        self._nlp = spacy.load('en_core_web_md')
        super().__init__(list(self._nlp.vocab.strings))

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
                self._model_path = dwnldr.load('word2vec-google-news-300', return_path=True)
                print(f'MODEL PATH: {self._model_path}')
            else:
                sys.exit(1)

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
    result = x.build_feature_vectors(np.array([
        "on approach; captain (pilot flying) called for flaps 8; and i positioned flap lever to 8 position. an amber eicas message 'flaps fail' annunciated with a master caution with the flaps failed at the 0 degree position.",
        'during climb to 17;000 feet the first officer noticed subtle propeller fluctuations.',
        'i was conducting ojt with a developmental that has 1 r-side and all d-sides.',
        'ocean west and offshore west/central were combined.',
        'during climb to 17;000 ft the first officer noticed subtle propeller fluctuations.',
    ]), 4)

    print(result)
