class WordIndexMap(object):

    def __init__(self, data_list):
        self.index2word = data_list
        self.word2index = {}
        for i_iter, item in enumerate(self.index2word):
            # assert item not in self.word2index.keys()
            self.word2index[item] = i_iter

    def get_word_by_index(self, index):
        # assert index < len(self.index2word)
        return self.index2word[index]

    def get_index_by_word(self, word):
        # assert word in self.word2index.keys()
        return self.word2index[word]

    def get_word2index(self):
        return self.word2index

