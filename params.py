import os

class Run_Params:
    def __init__(self):

        self.mimic0_movie1_wiki2 = 2  # 0 for mimic and 1 for movie and 2 for wiki
        self.onehot0_embedding = 8   # 0 for onehot and 1-8 cbow/skipgram/fastCbow/fastSkipgram/glove/lda based skipgram/skipgram+SGNS/skipgram||SGNS
        self.lm_lda_l2 = 1  # 0 for lstm+ldareg; 1 for lstm+l2; 2 for mlp+ldareg; 3 for mlp+l2
        self.gpu = '2'
        self.save = True
        self.save_path = "/home1/yk/experiments_TKDE/major_revision/"
        self.save_name = "movie_wiki_LSTM_SG_concat_ldaSGNS_l2_2layer"

    def show(self):
        print("Parameters of this run used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")

    def save_self(self, file_path, file_name):
        with open(os.path.join(file_path, file_name), 'a+') as f:
            f.write("----------Parameters of this run:------------\n")
            for item in self.__dict__.items():
                f.write("%s: %s\n" % item)
            f.write("-----------------------------------------------\n")
runP = Run_Params()

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

    def save_self(self, file_path, file_name):
        with open(os.path.join(file_path, file_name), 'a+') as f:
            f.write("----------Parameters of mimic data:------------\n")
            for item in self.__dict__.items():
                f.write("%s: %s\n" % item)
            f.write("-----------------------------------------------\n")

MIMICP = MimicParams()
# MIMICP.show()


class MovieParams:
    def __init__(self):

        self.movie_data_path = "/home1/yk/Movie_Review_data/"
        self.feature_index_file = self.movie_data_path + "new_word2index.pkl"   # for training lda
        self.seq_num = 25
        self.train_percent = 0.8
        self.num_words = 5229

    def show(self):
        print("Parameters of movie data used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")

    def save_self(self, file_path, file_name):
        with open(os.path.join(file_path, file_name), 'a+') as f:
            f.write("----------Parameters of movie data:------------\n")
            for item in self.__dict__.items():
                f.write("%s: %s\n" % item)
            f.write("-----------------------------------------------\n")
MOVIEP = MovieParams()
# MP.show()


class EmbeddingParams:
    def __init__(self):
        names = ['embedding', 'embedding_skipgram', 'fasttext', 'fasttext_skipgram', 'glove', 'lda_sgns', 'sg_add_sgns', 'sg_cancat_sgns']
        self.embedding_type = names[runP.onehot0_embedding - 1]   # embedding/embedding_skipgram/fasttext/fasttext_skipgram/glove
        self.veclen = 500
        # if self.embedding_type == 'sg_cancat_sgns':
        #     self.veclen = self.veclen * 2
        self.window = 50

    def show(self):
        print("Parameters of embedding used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")

    def save_self(self, file_path, file_name):
        with open(os.path.join(file_path, file_name), 'a+') as f:
            f.write("----------Parameters of embedding:------------\n")
            for item in self.__dict__.items():
                f.write("%s: %s\n" % item)
            f.write("-----------------------------------------------\n")

EMBEDP = EmbeddingParams()


# MP.show()


class LSTMParams:
    def __init__(self):
        self.hidden_size = 128
        self.num_classes = 1
        self.num_layers = 2
        self.drop_out = 0  # 0 for no drop out

        self.num_epochs = 600
        self.batchsize = 10
        self.learning_rate = 0.001
        self.weight_decay = 0.00001
        self.use_gpu = True

    def show(self):
        print("Parameters of LSTM used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")

    def save_self(self, file_path, file_name):
        with open(os.path.join(file_path, file_name), 'a+') as f:
            f.write("----------Parameters of LSTM:------------\n")
            for item in self.__dict__.items():
                f.write("%s: %s\n" % item)
            f.write("-----------------------------------------------\n")
LSTMP = LSTMParams()
# LSTMP.show()


class MLPParams:
    def __init__(self):
        self.hidden_size = 128
        self.num_classes = 1
        self.num_layers = 2

        self.num_epochs = 600
        self.batchsize = 10
        self.learning_rate = 0.001
        self.weight_decay = 0.00001
        self.use_gpu = True

    def show(self):
        print("Parameters of MLP used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")

    def save_self(self, file_path, file_name):
        with open(os.path.join(file_path, file_name), 'a+') as f:
            f.write("----------Parameters of MLP:------------\n")
            for item in self.__dict__.items():
                f.write("%s: %s\n" % item)
            f.write("-----------------------------------------------\n")
MLPP = MLPParams()
# MLPP.show()



class LDAParams:
    def __init__(self):
        self.corpus_path = "/home1/yk/wikipedia_dataset/filter"#"/home1/yk/new_mimic_formal_data/"#"/home1/yk/Movie_Review_data/"#"/home1/yk/wikipedia_dataset/filter"##"
        self.mimic0_movie1_wiki2 = 2 # 0 for mimic; 1 for movie and 2 for wiki
        self.corpus_file = "selected_movie_review_docs4LDA.csv"#"selected_docs4LDA.csv"#"selected_movie_review_docs4LDA.csv"#
        self.num_topic = 50
        self.plsa = False
        self.corpus_percent = 1
        self.output_path = "/home1/yk/experiments_TKDE/major_revision/"#"/home1/yk/new_mimic_formal_data/"#"/home1/yk/Movie_Review_data/"#"/home1/yk/wikipedia_dataset/filter"##"

    def show(self):
        print("Parameters of LDA used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")

    def save_self(self, file_path, file_name):
        with open(os.path.join(file_path, file_name), 'a+') as f:
            f.write("----------Parameters of LDA:------------\n")
            for item in self.__dict__.items():
                f.write("%s: %s\n" % item)
            f.write("-----------------------------------------------\n")
LDAP = LDAParams()
# LDAP.show()


class LDARegParams:
    def __init__(self):

        self.param_lda = 1.0
        self.param_alpha = 1.0
        self.weight_decay = 0.0001

        self.paramuptfreq = 20
        self.ldauptfreq = 50

    def show(self):
        print("Parameters of LDAReg used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")

    def save_self(self, file_path, file_name):
        with open(os.path.join(file_path, file_name), 'a+') as f:
            f.write("----------Parameters of LDAReg:------------\n")
            for item in self.__dict__.items():
                f.write("%s: %s\n" % item)
            f.write("-----------------------------------------------\n")
ldaregP = LDARegParams()
# ldaregP.show()



