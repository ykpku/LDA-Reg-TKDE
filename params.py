
class MimicParams:
    def __init__(self):

        self.mimic_data_path = "/home1/yk/new_mimic_formal_data/"
        self.feature_index_file = self.mimic_data_path + "feature2index_seq.csv"  # for training lda
        self.seq_num = 9
        self.train_percent = 0.8

    def show(self):
        print("Parameters of mimic data used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")
MIMICP = MimicParams()
# MIMICP.show()


class MovieParams:
    def __init__(self):

        self.movie_data_path = "/home1/yk/Movie_Review_data/"
        self.feature_index_file = "new_word2index.pkl"   # for training lda
        self.seq_num = 25
        self.train_percent = 0.8
        self.num_words = 5229

    def show(self):
        print("Parameters of movie data used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")
MOVIEP = MovieParams()
# MP.show()


class EmbeddingParams:
    def __init__(self):
        names = ['embedding', 'embedding_skipgram', 'fasttext', 'fasttext_skipgram', 'glove']
        self.embedding_type = names[runP.onehot0_embedding - 1]   # embedding/embedding_skipgram/fasttext/fasttext_skipgram/glove
        self.veclen = 200
        self.window = 50

    def show(self):
        print("Parameters of embedding used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")

EMBEDP = EmbeddingParams()


# MP.show()


class LSTMParams:
    def __init__(self):
        self.hidden_size = 128
        self.num_classes = 80
        self.num_layers = 2
        self.drop_out = 0  # 0 for no drop out

        self.num_epochs = 2
        self.batchsize = 100
        self.learning_rate = 0.001
        self.weight_decay = 0.0
        self.use_gpu = True

    def show(self):
        print("Parameters of LSTM used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")
LSTMP = LSTMParams()
# LSTMP.show()


class MLPParams:
    def __init__(self):
        self.hidden_size = 128
        self.num_classes = 80
        self.num_layers = 2

        self.num_epochs = 2
        self.batchsize = 100
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.use_gpu = True

    def show(self):
        print("Parameters of MLP used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")
MLPP = MLPParams()
# MLPP.show()


class LDAParams:
    def __init__(self):
        self.cprpus_path = "/home1/yk/new_mimic_formal_data/"
        self.mimic0_movie1 = 0  # 0 for mimic and 1 for movie
        self.corpus_file = "selected_docs4LDA.csv"
        self.num_topic = 50
        self.plsa = False

    def show(self):
        print("Parameters of LDA used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")
LDAP = LDAParams()
# LDAP.show()


class LDARegParams:
    def __init__(self):

        self.param_lda = 1.0
        self.param_alpha = 1.0
        self.weight_decay = 0.0

        self.paramuptfreq = 20
        self.ldauptfreq = 50

    def show(self):
        print("Parameters of LDAReg used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")
ldaregP = LDARegParams()
# ldaregP.show()


class Run_Params:
    def __init__(self):

        self.mimic0_movie1 = 0  # 0 for mimic and 1 for movie
        self.onehot0_embedding = 0   # 0 for onehot and 1-5 cbow/skipgram/fastCbow/fastSkipgram/glove
        self.lm_lda_l2 = 0  # 0 for lstm+ldareg; 1 for lstm+l2; 2 for mlp+ldareg; 3 for mlp+l2

        self.save = True
        self.save_path = "/home1/yk/experiments_TKDE/"
        self.save_name = "test2epoch"

    def show(self):
        print("Parameters of this run used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")
runP = Run_Params()