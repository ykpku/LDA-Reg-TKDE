from __future__ import print_function
import gensim
import sys
from os import path
import smart_open

sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])
Path = path.join(path.split(path.split(path.abspath(path.dirname(__file__)))[0])[0], 'medical_data')


def glove2word2vec(glove_vector_file, output_model_file):
    """Convert GloVe vectors into word2vec C format"""

    def get_info(glove_file_name):
        """Return the number of vectors and dimensions in a file in GloVe format."""
        with smart_open.smart_open(glove_file_name) as f:
            num_lines = sum(1 for line in f)
        with smart_open.smart_open(glove_file_name) as f:
            num_dims = len(f.readline().split()) - 1
        return num_lines, num_dims

    def prepend_line(infile, outfile, line):
        """
        Function to prepend lines using smart_open
        """
        with smart_open.smart_open(infile, 'rb') as old:
            with smart_open.smart_open(outfile, 'wb') as new:
                new.write(str(line.strip()) + "\n")
                for line in old:
                    new.write(line)
        return outfile

    num_lines, dims = get_info(glove_vector_file)
    gensim_first_line = "{} {}".format(num_lines, dims)
    model_file = prepend_line(glove_vector_file, output_model_file, gensim_first_line)
    # Demo: Loads the newly created glove_model.txt into gensim API.
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False) #GloVe Model

    return model