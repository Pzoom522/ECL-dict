
import os
import re

from .utils import Utils

class FilesIO:

    def load_corpus(path, max_document_size=10000):
        splitter = re.compile("\s")
        lines  = open(path, "r").read().splitlines()
        tokens = [splitter.split(e) for e in lines]
        documents = Utils.flatten([list(Utils.chunks(r, max_document_size)) for r in tokens])
        return documents

    def path(name):
        return os.path.join(os.path.dirname(__file__), "datasets", name)
