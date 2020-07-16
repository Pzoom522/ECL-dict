
class WindowWeights:

    def create_window(left, right, weighter):
        # creates a function that yields a tuple of words, context and weight
        def window(document):
            doc_len = len(document)
            for iW, word in enumerate(document):
                for i in reversed(range(1, min(left, iW + 1))):
                    yield weighter(word, document[iW - i], i, left)
                for i in range(1, min(right, doc_len - iW)):
                    yield weighter(word, document[iW + i], i, right)

        def window_size(document):
            l1 = left - 1
            r1 = right - 1
            doc_len = len(document)
            size = doc_len * (l1 + r1) - (l1 * (l1 + 1)) / 2 - (r1 * (r1 + 1)) / 2
            return int(size)

        return window, window_size

    def weight_harmonic(word, context, dist, windowSize):
        # the harmonic weighing
        weight = 1.0 / dist
        return (word, context, weight)

    def weight_word2vec(word, context, dist, windowSize):
        # The word2vec weighing
        # In the paper, the word2vec weight is written as
        #      weight = (1.0 * dist) / windowSize
        # But that makes no sens to have a bigger weight for distant words,
        # so I inversed the formula
        weight = 1.0 * windowSize / dist
        return (word, context, weight)
