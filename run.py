import sys
from os import path

sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

from LDA_Reg import lstm_ldareg, mlp_ldareg
from baselines import lstm_l2, mlp_l2
from load_data import load_mimic_data, load_movie_data
from params import runP
from pre_models import lda
from com.save import save_results

def run():
    if runP.onehot0_embedding == 0:
        if runP.mimic0_movie1 == 0:
            train_x, train_y, test_x, test_y = load_mimic_data.reload_mimic_seq()
        else:
            train_x, train_y, test_x, test_y = load_movie_data.reload_movie_seq()
    else:
        if runP.mimic0_movie1 == 0:
            train_x, train_y, test_x, test_y = load_mimic_data.reload_mimic_embedding()
        else:
            train_x, train_y, test_x, test_y = load_movie_data.reload_movie_embedding()

    if runP.lm_lda_l2 == 0:
        lda_tool = lda.LdaTools()
        net, sita, result_epoch, time_epochs = lstm_ldareg.train(train_x, train_y, test_x, test_y, lda_tool)
    elif runP.lm_lda_l2 == 1:
        net, sita, result_epoch, time_epochs = lstm_l2.train(train_x, train_y, test_x, test_y)
    elif runP.lm_lda_l2 == 2:
        lda_tool = lda.LdaTools()
        net, sita, result_epoch, time_epochs = mlp_ldareg.train(train_x, train_y, test_x, test_y, lda_tool)
    else:
        net, sita, result_epoch, time_epochs = mlp_l2.train(train_x, train_y, test_x, test_y)

    if runP.save:
        save_results(net, sita, result_epoch, time_epochs, runP.save_path, runP.save_name)



run()

