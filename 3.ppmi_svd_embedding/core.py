"""
.. module:: argparse_actions
"""

import bz2
import pickle
import numpy as np
import pandas as pd
import multiprocessing

from scipy.sparse import vstack
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from collections import OrderedDict, Counter
from tqdm import tqdm, notebook 

from .utils import Utils
from .window import WindowWeights
from .temporary_array import TemporaryArray, NamedSparseArray


class svd2vec:
    """
    The representation of the documents words in a vector format.

    Parameters
    ----------
    documents : list of list of string
        The list of document, each document being a list of words
    size : int
        Maximum numbers of extracted features for each word
    min_count : int
        Minimum number of occurence of each word to be included in the model
    window : int or tuple of ints
        Window word counts for getting context of words.
        If an int is given, it's equivalent of a symmetric tuple (int, int).
    dyn_window_weight : WINDOW_WEIGHT_HARMONIC or WINDOW_WEIGHT_WORD2VEC
        The window weighing scheme.
    cds_alpha : float
        The context distribution smoothing constant that smooths the context
        frequency
    neg_k_shift : int
        The negative PMI log shifting
    eig_p_weight : float
        The eigenvalue weighting applied to the eigenvalue matrix
    nrm_type : string
        A normalization scheme to use with the L2 normalization
    sub_threshold : float
        A threshold for subsampling (diluting very frequent words). Higher value
        means less words removed.
    verbose : bool
        If True, displays progress during the init step
    workers : int
        The numbers of workers to use in parallel (should not exceed the
        available number of cores on the computer)
    """


    WINDOW_WEIGHT_HARMONIC = 0
    """The harmonic weighing scheme for context words *(1/5, 1/4, 1/3, 1/2, ...)*"""
    WINDOW_WEIGHT_WORD2VEC = 1
    """The word2vec weighing scheme for context words *(1/5, 2/5, 3/5, 4/5, ...)*"""


    NRM_SCHEME_NONE   = "none"
    NRM_SCHEME_ROW    = "row"
    NRM_SCHEME_COLUMN = "column"
    NRM_SCHEME_BOTH   = "both"
    NRM_SCHEMES = [NRM_SCHEME_NONE, NRM_SCHEME_ROW, NRM_SCHEME_COLUMN, NRM_SCHEME_BOTH]
    """Available normalization schemes"""

    MAX_CPU_CORES = -1

    def __init__(self,
                 documents,
                 size=150,
                 min_count=2,
                 window=10,
                 dyn_window_weight=WINDOW_WEIGHT_WORD2VEC,
                 cds_alpha=0.75,
                 neg_k_shift=1,
                 eig_p_weight=0,
                 nrm_type=NRM_SCHEME_ROW,
                 sub_threshold=1e-5,
                 verbose=False,
                 workers=MAX_CPU_CORES):

        # -------------
        # args checking
        # -------------

        # dyn weight
        if dyn_window_weight == svd2vec.WINDOW_WEIGHT_HARMONIC:
            window_weighter = WindowWeights.weight_harmonic
        elif dyn_window_weight == svd2vec.WINDOW_WEIGHT_WORD2VEC:
            window_weighter = WindowWeights.weight_word2vec
        else:
            raise ValueError(dyn_window_weight + " not implemented as a weighter")

        # window type
        if isinstance(window, int):
            window, window_size = WindowWeights.create_window(left=window, right=window, weighter=window_weighter)
        elif isinstance(window, tuple) and len(window) == 2 and all(map(lambda e: isinstance(e, int), window)):
            window, window_size = WindowWeights.create_window(left=window[0], right=window[1], weighter=window_weighter)
        else:
            raise ValueError("'" + window + "' not implemented as a window yielder")

        # normalization type
        if nrm_type not in svd2vec.NRM_SCHEMES:
            raise ValueError("'" + nrm_type + "' cannot be used in as a normalization method")

        # workers
        if isinstance(workers, int):
            if workers < 1:
                workers = multiprocessing.cpu_count()
        else:
            raise ValueError("'" + workers + "' is not a valid cpu count")

        # -----------
        # args saving
        # -----------

        self.workers       = workers
        self.min_count     = min_count
        self.size          = size
        self.window        = window
        self.window_size   = window_size
        self.cds_alpha     = cds_alpha
        self.sub_threshold = sub_threshold
        self.neg_k_shift   = neg_k_shift
        self.eig_p_weight  = eig_p_weight
        self.nrm_type      = nrm_type
        self.verbose       = verbose
        self.bar_offset    = 0

        # --------------------
        # document preparation
        # --------------------
        self.build_vocabulary(documents)
        self.subsampling()
        self.weighted_count_matrix_file = self.skipgram_weighted_count_matrix()
        self.clean_instance_variables()

        # ---------------
        # pmi computation
        # ---------------
        self.pmi = self.sppmi_matrix(self.pmi_matrix())

        # ---------------
        # svd computation
        # ---------------

        self.svd_w, self.svd_c = self.svd()

        # -------
        # closing
        # -------
        # weighted_count_matrix_file was not a simple numpy matrix at a path to
        # a memmap of a numpy matrix. Now we can remove the temporary file
        self.weighted_count_matrix_file.close()

    #####
    # Building informations matrices and variables to be used later
    #####

    def build_vocabulary(self, documents):
        bar = self.bar(desc="vocabulary building", total=9)
        self.documents = documents
        bar.update()
        self.all_words = Utils.flatten(self.documents)
        bar.update()
        self.d_size    = len(self.all_words)
        bar.update()

        self.terms_counts = Counter(self.all_words)
        bar.update()
        self.d_alpha = np.sum(np.power([self.terms_counts[c] for c in self.terms_counts], self.cds_alpha))
        bar.update()
        self.terms_counts_cds_powered = {word: self.d_alpha / np.power(self.terms_counts[word], self.cds_alpha) for word in self.terms_counts}
        bar.update()

        self.vocabulary = OrderedDict([(w, i) for i, (w, c) in enumerate(self.terms_counts.most_common())])
        bar.update()
        self.words = list(self.vocabulary.keys())
        bar.update()
        self.vocabulary_len = len(self.vocabulary)
        bar.update()
        bar.close()

    def subsampling(self):
        new_docs = []
        for document in self.bar(self.documents, "subsampling"):
            new_words = []
            for word in document:
                if self.terms_counts[word] < self.min_count:
                    continue
                word_frequency = 1.0 * self.terms_counts[word] / self.d_size
                prob = 1 - np.sqrt(self.sub_threshold / word_frequency)
                if not Utils.random_decision(prob):
                    # we keep the word
                    new_words.append(word)
            new_docs.append(new_words)
        self.build_vocabulary(new_docs)

    def bar(self, yielder=None, desc=None, total=None, offset=0, parallel=False):
        disable = not self.verbose
        notebook = Utils.running_notebook()
        if notebook and self.verbose and parallel:
            # solves a bug in jupyter notebooks
            # https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308
            print('\r', end='', flush=True)
        func   = notebook.tqdm_notebook if notebook else tqdm
        format = "{desc: <30} {percentage:3.0f}%  {bar}"
        return func(
            iterable=yielder,
            desc=desc,
            leave=False,
            total=total,
            disable=disable,
            position=offset,
            bar_format=format)

    def skipgram_weighted_count_matrix(self):
        file = TemporaryArray((self.vocabulary_len, self.vocabulary_len), np.dtype('float16'))
        matrix = file.load(erase=True)

        for document in self.bar(self.documents, "co-occurence"):
            for word, context, weight in self.window(document):
                # i_word    = self.vocabulary[word]
                # i_context = self.vocabulary[context]
                # matrix[i_word, i_context] += weight
                matrix[self.vocabulary[word], self.vocabulary[context]] += weight

        matrix.flush()
        del matrix
        return file

    def clean_instance_variables(self):
        # these two instances variables uses too much RAM, and it's not longer
        # useful
        delattr(self, "all_words")
        delattr(self, "documents")

        # we do not need the window function anymore, plus keeping it as an
        # instance variable will stop us from using joblib parallelisation
        # because this can not be saved as a pickle object
        delattr(self, "window")
        delattr(self, "window_size")

    def pmi_matrix(self):
        # pointwise mutal information

        slices = Utils.split(list(self.vocabulary), self.workers)
        pmi_name_list  = Parallel(n_jobs=self.workers)(delayed(self.pmi_parallized)(slice, i) for i, slice in enumerate(slices) if slice != [])
        pmi_array_list = [NamedSparseArray.from_name(array) for array in pmi_name_list]

        pmi = vstack([named_array.get_matrix() for named_array in pmi_array_list])

        [named_array.delete() for named_array in pmi_array_list]

        return pmi

    def pmi_parallized(self, slice, i):
        # returns a small matrix corresponding to the slice of words given (rows)
        # python processing api does not works with big arrays, so we store the array to the disk and we return it's name
        array = NamedSparseArray.new_one(shape=(len(slice), self.vocabulary_len), dtype=np.dtype('float64'))
        pmi   = array.get_matrix()

        self.weighted_count_matrix_offset = self.vocabulary[slice[0]]
        self.weighted_count_matrix = self.weighted_count_matrix_file.load(size=len(slice), start=self.weighted_count_matrix_offset)

        name = "pmi " + str(i + 1) + " / " + str(self.workers)

        for i_word, word in enumerate(self.bar(slice, desc=name, offset=i, parallel=True)):
            for context in self.vocabulary:
                i_context = self.vocabulary[context]

                n_wc = self.weighted_count_matrix[self.vocabulary[word] - self.weighted_count_matrix_offset, self.vocabulary[context]]
                n_w  = self.terms_counts[word]
                n_c_powered = self.terms_counts_cds_powered[context]

                if n_wc == 0:
                    continue

                frac = n_wc / (n_w * n_c_powered)

                pmi[i_word, i_context] = np.log(frac)

        array.save()
        del self.weighted_count_matrix

        return array.name

    def sppmi_matrix(self, pmi):
        # shifted positive pointwise mutal information
        shift = np.log(self.neg_k_shift)
        pmi.data = np.array([v - shift for v in pmi.data])
        return pmi

    def svd(self):
        bar = self.bar(desc="svd", total=5)
        modified_k = min(self.size, self.pmi.shape[0] - 1)
        bar.update()
        u, s, v = svds(self.pmi, k=modified_k)
        bar.update()

        w_svd_p = u * np.power(s, self.eig_p_weight)
        bar.update()
        c_svd   = v.T
        bar.update()

        w_svd_p = self.normalize(w_svd_p, self.nrm_type)
        bar.update()
        bar.close()

        return w_svd_p, c_svd

    def normalize(self, matrix, nrm_type):
        if nrm_type == svd2vec.NRM_SCHEME_NONE:
            return matrix
        if nrm_type == svd2vec.NRM_SCHEME_ROW:
            axis = 1 if matrix.ndim is not 1 else 0
            return matrix / np.linalg.norm(matrix, axis=axis, keepdims=True)
        if nrm_type == svd2vec.NRM_SCHEME_COLUMN:
            return matrix / np.linalg.norm(matrix, axis=0, keepdims=True)
        if nrm_type == svd2vec.NRM_SCHEME_BOTH:
            return matrix / np.linalg.norm(matrix, keepdims=True)
        raise ValueError("Normalization '" + nrm_type + "' error")

    #####
    # I/O
    #####

    def save(self, path):
        """
        Saves the svd2vec object to the given path.

        Parameters
        ----------
        path : string
            The file path to write the object to. The directories should exists.
        """

        with bz2.open(path, "wb") as file:
            pickle.dump(self, file)

    def load(path):
        """
        Load a previously saved svd2vec object from a path.

        Parameters
        ----------
        path : string
            The file path to load the object from.

        Returns
        -------
        svd2vec
            A new `svd2vec` object
        """

        with bz2.open(path, "rb") as file:
            return pickle.load(file)

    def save_word2vec_format(self, path):
        """
        Saves the word vectors to a path using the same format as word2vec.
        The file can then be used by other modules or libraries able to load
        word2vec vectors.

        Parameters
        ----------
        path : string
            The file path to write the object to. The directories should exists.
        """

        with open(path, "w") as f:
            print(str(self.vocabulary_len) + " " + str(self.size), file=f)
            for word in self.vocabulary:
                values = " ".join(["{:.6f}".format(e) for e in self.vector_w(word)])
                print(word + " " + values, file=f)

    #####
    # Getting informations
    #####

    def pmi(self, word, context):
        n_wc = self.weighted_count_matrix[self.vocabulary[word] - self.weighted_count_matrix_offset, self.vocabulary[context]]
        n_w  = self.terms_counts[word]
        n_c_powered = self.terms_counts_cds_powered[context]

        p_wc = n_wc / self.d_size
        p_w  = n_w  / self.d_size
        p_c  = n_c_powered

        frac = p_wc / (p_w * p_c)

        if frac == 0:
            return None  # should in theory be -np.inf, but we're doing the PPMI directly 
        return np.log(frac)

    def cosine_similarity(self, wx, cx, wy, cy):
        # Unused : you should not use the cosine function itself, but the dot
        # product (since we normalized the vectors, it's the same, but way
        # faster)
        # Compute the cosine similarity of x (word x and context x) and y (word
        # y and context y)
        wxcx = wx + cx
        wycy = wy + cy
        top = np.dot(wxcx, wycy)
        bot = np.sqrt(np.dot(wxcx, wxcx)) * np.sqrt(np.dot(wycy, wycy))
        return top / bot


    def similarity(self, x, y):
        """
        Computes and returns the cosine similarity of the two given words.

        Parameters
        ----------
        x : string
            The first word to compute the similarity
        y : string
            The second word to compute the similarity

        Returns
        -------
        float
            The cosine similarity between the two words

        Warning
        -------
        The two words ``x`` and ``y`` should have been trainned during the
        initialization step.
        """
        wx = self.vector_w(x)
        wy = self.vector_w(y)
        sim = np.dot(wx, wy)
        return sim

    def distance(self, x, y):
        """
        Computes and returns the cosine distance of the two given words.

        Parameters
        ----------
        x : string
            The first word to compute the distance
        y : string
            The second word to compute the distance

        Returns
        -------
        float
            The cosine distance between the two words

        Raises
        ------
        ValueError
            If either x or y have not been trained during the initialization step.

        Warning
        -------
        The two words ``x`` and ``y`` should have been trained during the
        initialization step.
        """

        sim = self.similarity(x, y)
        return 1 - sim

    def most_similar(self, positive=[], negative=[], topn=10):
        """
        Computes and returns the most similar words from those given in positive
        and negative.

        Parameters
        ----------
        positive : list of string or string
            Each word in positive will contribute positively to the output words
            A single word can also be passed to compute it's most similar words.
        negative : list of string
            Each word in negative will contribute negatively to the output words
        topn : int
            Number of similar words to output

        Returns
        -------
        list of ``(word, similarity)``
            Each tuple is a similar word with it's similarity to the given word.

        Raises
        ------
        ValueError
            If the no input is given in both positive and negative
        ValueError
            If some words have not been trained during the initialization step.

        Warning
        -------
        The input words should have been trained during the
        initialization step.

        """

        # Output the most similar words for the given positive and negative
        # words. topn limits the number of output words
        if isinstance(positive, str):
            positive = [positive]
        if not isinstance(positive, list) or not isinstance(negative, list):
            raise ValueError("Positive and Negative should be a list of words inside the vocabulary")
        if positive == [] and negative == []:
            raise ValueError("Cannot get the most similar words without any positive or negative words")

        positives_w = [self.vector_w(x) for x in positive]
        negatives_w = [self.vector_w(x) for x in negative]

        if positive and negative:
            all_w = np.concatenate([positives_w, -np.array(negatives_w)])
        elif positive:
            all_w = np.array(positives_w)
        elif negative:
            all_w = np.array(negatives_w)

        current_w = self.normalize(all_w.mean(axis=0), self.nrm_type)

        exclude_words = np.concatenate([positive, negative])

        distances = np.dot(self.svd_w, current_w)
        similarities = np.array([self.words, distances], dtype=object).T
        sorted   = similarities[(-similarities[:, 1]).argsort()]
        selected = sorted[~np.in1d(sorted[:, 0], exclude_words)][:topn].tolist()

        return selected

    def analogy(self, exampleA, answerA, exampleB, topn=10):
        """
        Returns the topn most probable answers to the analogy question "exampleA
        if to answerA as exampleB is to ?"

        Parameters
        ----------
        exampleA : string
            The first word to "train" the analogy on
        answerA : string
            The second word to "train" the analogy on
        exampleB : string
            The first word to ask the answer

        Returns
        -------
        list of (word, similarity)
            Each word and similarity is a probable answer to the analogy

        Raises
        ------
        ValueError
            If some words have not been trained during the initialization step.

        Warning
        -------
        The three input words should have been trained during the
        initialization step.

        """
        # returns answerB, ie the answer to the question
        # exampleA is to answerA as exampleB is to answerB
        return self.most_similar(positive=[exampleB, answerA], negative=[exampleA], topn=topn)

    def vectors(self, word):
        if word in self.vocabulary:
            i_word = self.vocabulary[word]
            w = self.svd_w[i_word]
            c = self.svd_c[i_word]
            return w, c
        else:
            raise ValueError("Word '" + word + "' not in the vocabulary")

    def vector_w(self, word):
        return self.get_vector(word, self.svd_w)

    def vector_c(self, word):
        return self.get_vector(word, self.svd_c)

    def get_vector(self, word, dicti):
        if word in self.vocabulary:
            i_word = self.vocabulary[word]
            return dicti[i_word]
        else:
            raise ValueError("Word '" + word + "' not in the vocabulary")

    #####
    # Evaluation
    #####

    def evaluate_word_pairs(self, pairs, delimiter='\t'):
        """
        Evaluates the model similarity using a pairs file of human judgments
        of similarities.

        Parameters
        ----------
        pairs : string
            A filepath of a csv file. Lines starting by '#' will be ignored.
            The first and second column are the words. The third column is the
            human made similarity.
        delimiter : string
            The delimiter of the csv file

        Returns
        -------
        tuple
            The first value is the pearson coefficient (1.0 means the model is
            very good according to humans, 0.0 it's very bad). The second value
            is the two-tailed p-value.

        """
        file = Utils.parse_csv(pairs, delimiter)
        x = []
        y = []
        for row in file:
            w1 = row[0]
            w2 = row[1]
            hsim = float(row[2])
            if w1 not in self.vocabulary or w2 not in self.vocabulary:
                continue
            msim = self.similarity(w1, w2)
            x.append(hsim)
            y.append(msim)
        pearson, p_value, low, high = Utils.confidence_pearson(np.array(x), np.array(y))
        return pearson, p_value, (low, high)

    def evaluate_word_analogies(self, analogies, section_separator=":"):

        selected_analogies = []
        with open(analogies, "r") as file:
            for line in file.read().splitlines():
                if line.startswith(section_separator) or line.startswith('#'):
                    continue
                words = line.lower().split(" ")
                if any([w not in self.vocabulary for w in words]):
                    continue
                selected_analogies.append(words)

        total   = len(selected_analogies)
        correct = 0

        for w1, w2, w3, w4 in self.bar(selected_analogies, "analogies computing"):
            if self.analogy(w1, w2, w3, topn=1)[0][0] == w4:
                correct += 1
        return (1.0 * correct) / total

    #####
    # Debug
    #####

    def display_matrix(self, matrix, vocabulary=None):
        if vocabulary is None:
            vocabulary = self.vocabulary
        z = {a: e for e, a in vocabulary.items()}
        v = [z[i] for i in sorted(vocabulary.values())]
        df = pd.DataFrame(matrix.toarray(), columns=v, index=v)
        df = df.applymap(lambda x: '{:4.2f}'.format(x) if x != 0 else "")
        print(df)

    def display_similarity(self, word1, word2):
        sim = self.similarity(word1, word2)
        print(word1, " & ", word2, sim)

    def display_most_similar(self, positive=[], negative=[]):
        sims = self.most_similar(positive=positive, negative=negative)
        print(positive, negative, sims)
