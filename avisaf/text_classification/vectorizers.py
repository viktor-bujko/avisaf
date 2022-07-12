#!/usr/bin/env python3

import re
import sys
import json
import lzma
import spacy
import pickle
import logging
import numpy as np
from pathlib import Path
from spacy.language import Doc
from spacy.lang.en import STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import utils
import gensim.downloader as dwnldr
from gensim.models import Doc2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument


logger = logging.getLogger("avisaf_logger")
# words which could play an important role in vector representation
stops_overrides = ["above", "below", "first"]

for not_stop_word in stops_overrides:
    STOP_WORDS.remove(not_stop_word)


class VectorizerFactory:
    @staticmethod
    def create_vectorizer(vect: str):
        default = TfIdfAsrsReportVectorizer()
        if not vect or vect == "default":
            logger.info(f"Using default vectorizer: {default}")
            return default

        available_vectorizers = {
            "tfidf": lambda: default,
            "spacyw2v": lambda: SpaCyWord2VecAsrsReportVectorizer(),
            "googlew2v": lambda: GoogleNewsWord2VecAsrsReportVectorizer(),
            "d2v": lambda: Doc2VecAsrsReportVectorizer()
        }

        vectorizer = available_vectorizers.get(vect, default)()
        logger.info(f"Chosen vectorizer: {vectorizer}")
        return vectorizer


class AsrsReportVectorizer:

    def vectorize_texts(self, extractor, return_vectors: bool = True):
        narrative_label = "Report 1_Narrative"
        narratives = extractor.extract_data([narrative_label])[narrative_label]

        if return_vectors:
            return self._build_feature_vectors(narratives)
        else:
            return narratives

    def _build_feature_vectors(self, texts: np.ndarray) -> np.ndarray:
        """
        :param texts:
        :type texts: np.ndarray
        :return: Feature vectors for given texts.
        :rtype: np.ndarray
        """
        pass

    @staticmethod
    def preprocess(texts: np.ndarray):
        logger.debug("Started preprocessing")
        preprocessed = []
        logger.debug("Loading language model")
        nlp = spacy.load("./models/ner/ner_avisaf.model")
        logger.debug("Loading done")
        texts = map(lambda txt: str(txt), texts)  # converting numpy string for

        with Path("config", "aviation_glossary.json").open("r") as glossary_file:
            logger.debug("Loading terminology glossary")
            abbreviations_glossary = dict(json.load(glossary_file))
            logger.debug("Glossary loaded")

        batch_size = 1024
        for idx, doc in enumerate(nlp.pipe(
                texts,
                batch_size=batch_size,
                disable=[],
                n_process=4)
        ):
            if idx > 0 and idx % batch_size == 0:
                logger.debug(f"Lemmatized { int((idx / batch_size) * batch_size) } texts.")
            doc_lemmas = []
            for token in doc:
                token_base_form = token.lemma_
                if token.is_stop:
                    # skipping / ignoring stop words
                    continue
                if token.text.startswith("ZZZ"):
                    token_base_form = "airport"
                if token.ent_type_ == "NAV_WAYPOINT":
                    token_base_form = "waypoint"
                if token.ent_type_ == "ABBREVIATION":
                    # searching for full version of abbreviation in the glossary
                    abbr_explained = abbreviations_glossary.get(token.text, {"full": token.text})
                    # get full version of abbreviation if exists, otherwise use abbreviation without change
                    token_base_form = abbr_explained.get("full")

                # replacing token text by its lemma while preserving original whitespace
                doc_lemmas.append(token_base_form.lower() + token.whitespace_)
            lemmatized = "".join(doc_lemmas)

            lemmatized = re.sub(r"(\d{1,2});(\d{1,3})", r"\1,\2", lemmatized)  # ; separated numbers - usually altitude
            lemmatized = re.sub(r"fl;?(\d{2,3})", r"flight level \1", lemmatized)  # flight level representation
            lemmatized = re.sub(r"runway|rwy \d{1,2}[rcl]?", r"runway", lemmatized)  # runway identifiers
            lemmatized = re.sub(r"tx?wys?", "taxiway", lemmatized)
            lemmatized = re.sub(r"twrs?[^a-z]", "tower", lemmatized)
            lemmatized = re.sub(r"([a-z\d]+\.){2,}[a-z\d]*", "", lemmatized)  # removing words with several dots
            lemmatized = re.sub(r"(air)?spds?", "speed", lemmatized)
            lemmatized = re.sub(r"lndgs?", "landing", lemmatized)

            preprocessed.append(lemmatized)
        logger.debug("Ended preprocessing")

        return preprocessed

    def get_params(self) -> dict:
        pass


class TfIdfAsrsReportVectorizer(AsrsReportVectorizer):
    def __init__(self):

        self.transformer_name = "tfidf"
        self._transformer = TfidfVectorizer(
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 2),
            max_features=100_000,
        )
        self._model_dir = Path("vectors", self.transformer_name)
        self._model_path = Path(self._model_dir, "tfidf_pipeline.model")
        self._pipeline = Pipeline([
            ("tfidf", self._transformer),
            ("scaler", StandardScaler(with_mean=False)),
            ("reductor", TruncatedSVD(n_components=300)),
        ])

    def _build_feature_vectors(self, texts: np.ndarray):
        logger.debug("Started vectorization")

        texts = self.preprocess(texts)

        if not self._model_path.exists():
            if not self._model_dir.exists():
                self._model_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("tfidf training started")
            texts_vectors = self._pipeline.fit_transform(texts)
            logger.debug("tfidf training finished")

            with lzma.open(self._model_path, "wb") as pipe:
                logger.info("Serializing tfidf pipeline")
                pickle.dump(self._pipeline, pipe)
                logger.info("Serialization successful")
        else:
            with lzma.open(self._model_path, "rb") as pipe:
                self._pipeline = pickle.load(pipe)
            texts_vectors = self._pipeline.transform(texts)

        logger.debug("Ended vectorization")
        return texts_vectors

    def get_params(self):
        return {
            "name": "TfIdfVectorizer",
            "vectorizer": self.transformer_name,
            "encoding": self._transformer.encoding,
            "decode_error": self._transformer.decode_error,
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
        self._model_path = str(Path("vectors", "doc2vec", "doc2vec.model"))

    def train_new_model(self, tagged_docs):
        model = Doc2Vec(
            vector_size=300,
            epochs=40,
            min_count=5,
            dm=1,
            dm_mean=0,
            dm_concat=0,
            dbow_words=0,
            negative=8,
            workers=3,
            window=4,
            alpha=0.001,
            min_alpha=0.0001,
        )
        logger.debug(f"Estimated memory: {model.estimate_memory()}")
        logger.debug("Starting vocabulary build")
        model.build_vocab(documents=tagged_docs)
        logger.debug("Ended vocabulary build")

        logger.debug("Started doc2vec model training")
        model.train(
            documents=tagged_docs,
            total_examples=model.corpus_count,
            epochs=model.epochs,
        )
        logger.debug("Ended doc2vec model training")
        model.save(self._model_path)

        return model

    def _build_feature_vectors(self, texts: type(np.ndarray)):

        try:
            model = Doc2Vec.load(self._model_path, mmap="r")
        except FileNotFoundError:
            model = None

        tagged_docs = []
        texts = np.array(self.preprocess(texts))

        for idx, text in enumerate(texts):
            tokens = utils.simple_preprocess(text)
            tagged_docs.append(TaggedDocument(words=tokens, tags=[idx]))

        if model is None:
            logger.warning("No trained doc2vec model to infer text vectors from. Training a new one.")
            model = self.train_new_model(tagged_docs)

        doc2veced = np.zeros(shape=(texts.shape[0], model.vector_size))
        if len(tagged_docs) != texts.shape[0]:
            raise ValueError("Incorrect dimensions of tagged_docs and texts on input")

        doc2veced_count = 0
        for idx, tagged_doc in enumerate(tagged_docs):
            doc2veced_count += 1
            if doc2veced_count % 1000 == 0:
                logger.debug(f"Number of vectorized texts: {doc2veced_count}")
            doc2veced[idx] = model.infer_vector(tagged_doc.words, epochs=6)
        return doc2veced

    def get_params(self):
        return {
            "name": "Doc2Vec",
            "vectorizer": "d2v",
            "path": self._model_path
        }


class Word2VecAsrsReportVectorizer(AsrsReportVectorizer):
    def __init__(self, vectors=None):
        self._nlp = spacy.load("en_core_web_md")
        self._vectors = vectors

    def _build_feature_vectors(self, texts: np.ndarray) -> np.ndarray:
        logger.debug("Started vectorization")

        texts = self.preprocess(texts)

        doc_vectors = []
        for doc_vector_batch in self._generate_vectors(texts, self._vectors, 256):
            doc_vectors.append(doc_vector_batch)
            print("===========================")

        result = np.concatenate(doc_vectors, axis=0)
        logger.debug(f"Vectorized {result.shape[0]} texts")

        return result

    def _generate_vectors(self, texts, vectors, batch: int = 100):
        oov = set()
        doc_vectors = []
        for doc in self._nlp.pipe(texts, disable=["ner"], batch_size=batch):
            lemmas = []
            for token in doc:
                if token.pos_ == "PRON" and token.text in vectors:
                    # taking "we", "he" etc instead of "-PRON-"
                    lemmas.append(token.text)
                    continue
                if (
                    token.is_punct
                    or token.is_stop
                    or str(token.text).isspace()
                    or not str(token.text)
                ):
                    # ignoring the punctuation and empty/whitespace tokens
                    # ignoring the stop words, but **after** taking the pronouns
                    continue
                if token.like_num:
                    # replacing all numbers by common tag
                    lemmas.append("number")
                    continue
                if token.lemma_ not in vectors:
                    # trying to identify some common mistakes
                    words = re.sub(
                        r"([a-z]*)[?!\-.]([a-z]*)", r"\1 \2", token.text
                    ).split()  # [?!\-.] separated words replaced by space
                    for word in words:
                        if word not in vectors:
                            continue
                        lemmas.append(word)

                    # ignoring word which don't have vector representation
                    if token.text not in oov:
                        logger.warning(f'Word "{token.text}" with lemma "{token.lemma_}" not in vocabulary')
                        oov.add(token.text)
                    continue

                lemmas.append(token.lemma_)

            doc_vector = self._get_doc_vector(lemmas)  #
            doc_vectors.append(doc_vector)

        yield np.array(doc_vectors)

        # with open('out_of_vocab_20210421.txt', 'w') as oov_file:
        #    logger.debug("Saving unused words")
        #    print(*sorted(list(oov)), sep='\n', file=oov_file)

    def _get_doc_vector(self, lemmas):
        pass


class SpaCyWord2VecAsrsReportVectorizer(Word2VecAsrsReportVectorizer):
    def __init__(self):
        logger.debug("Loading spacy model")
        self._nlp = spacy.load("en_core_web_md")
        super().__init__(list(self._nlp.vocab.strings))

    def _get_doc_vector(self, lemmas):
        return Doc(self._nlp.vocab, words=lemmas).vector * (len(lemmas) / 500)

    def get_params(self):
        return {
            "name": "SpaCyWord2Vec",
            "vectorizer": "spacyw2v"
        }


class GoogleNewsWord2VecAsrsReportVectorizer(Word2VecAsrsReportVectorizer):
    def __init__(self):
        logger.debug(Path().absolute())
        self._model_path = Path(".", "vectors", "gensim-data", "word2vec-google-news-300", "word2vec-google-news-300.gz")
        print(self._model_path)
        if not self._model_path.exists():
            logger.warning("Pre-trained Google News model used for vectorization has not been found.")
            if input("Do you want to download and unzip the model (1.5 Gb zipped size)? (y/N) ").lower() == "y":
                logger.debug("(down)LOADING")
                self._model_path = dwnldr.load("word2vec-google-news-300", return_path=True)
                print(f"MODEL PATH: {self._model_path}")
            else:
                sys.exit(1)

        print("Loading large Google News model. May take a while.")
        self._model = KeyedVectors.load_word2vec_format(self._model_path, binary=True)
        super().__init__(self._model.wv)

    def _get_doc_vector(self, lemmas):
        return np.mean(self._model[lemmas] * (len(lemmas) / 500), axis=0)

    def get_params(self):
        return {
            "name": "GoogleNewsWord2Vec",
            "vectorizer": "googlew2v",
            "model_path": str(self._model_path)
        }


# class FastTextAsrsReportVectorizer(Word2VecAsrsReportVectorizer):
#     def __init__(self):
#         model_name = "cc.en.300.bin"
#         self._nlp = spacy.load("en_core_web_md")
#         self._vectors = fasttext.load_model(model_name)
#         self._model_path = str(Path("vectors", "fasttext_vectors", "fasttext.model"))
#         super().__init__(vectors=self._vectors)
#
#     def _build_feature_vectors(self, texts: np.ndarray) -> np.ndarray:
#         logger.debug("Started vectorization")
#
#         try:
#             model = fasttext.load_model(self._model_path)
#         except ValueError:
#             model = None
#
#         texts = self.preprocess(texts)
#
#         if model is None:
#             _, filename = tempfile.mkstemp(text=True)
#             with open(filename, "w") as f:
#                 print(*texts, sep="\n", file=f)
#             model = fasttext.train_unsupervised(filename, model="skipgram", dim=300, ws=8, epoch=25, minCount=3)
#
#             self._vectors = model
#             model.save_model(self._model_path)
#
#         doc_vectors = []
#         for doc_vector_batch in self._generate_vectors(texts, model, 256):
#             doc_vectors.append(doc_vector_batch)
#             print("===========================")
#
#         result = np.concatenate(doc_vectors, axis=0)
#         logger.debug(f"Vectorized {result.shape[0]} texts")
#
#         return result
#
#     def _generate_vectors(self, texts, vectors, batch: int = 100):
#         doc_vectors = []
#         for doc in self._nlp.pipe(texts, disable=["ner"], batch_size=batch):
#             lemmas = []
#             for token in doc:
#                 lemmas.append(vectors[token.text])
#
#             doc_vector = self._get_doc_vector(np.array(lemmas))
#             doc_vectors.append(doc_vector)
#
#         yield np.array(doc_vectors)
#
#     def _get_doc_vector(self, lemmas):
#         return np.mean(lemmas * (len(lemmas) / 500), axis=0)
#
#     def get_params(self):
#         return {
#             "name": "Fasttext",
#             "vectorizer": "fasttext",
#             "model_path": self._model_path
#         }
