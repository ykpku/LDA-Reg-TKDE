#  Author: Yang Kai
#  Date: 9/6/2017
#
#
# *************************************** #
import re
import nltk


from bs4 import BeautifulSoup



class Document2VecUtility(object):


    @staticmethod
    def review_to_word_list( review_text, remove_stopwords=True, stem_words=False, remove_html=False ):

        # Remove HTML
        if remove_html:
            review_text = BeautifulSoup(review_text, "lxml").get_text()
            pass

        # Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)

        # Convert words to lower case and split them
        words = review_text.lower().split()

        # Optionally remove stop words (false by default)
        if remove_stopwords:

            stops = set([])
            file_stop = open('en.txt')
            st_line = file_stop.readline()
            while st_line:
                stops.add(st_line.strip())
                st_line = file_stop.readline()
            words = [w for w in words if not w in stops]
        if stem_words:
            words = [nltk.PorterStemmer().stem(w) for w in words]

        return words

    @staticmethod
    def get_word_list(file_path, remove_stopwords=False, stem_words=False, remove_html=False):
        data = []
        line_num = 0
        with open(file_path, 'r') as f:
            line_list = f.readlines()
            for line in line_list:
                data.append(Document2VecUtility.review_to_word_list(line, remove_stopwords, stem_words,
                                                                    remove_html))
                # if line_num % 10000 == 0:
                    # print line_num
                    # print data[-1]
                    # print line
                line_num = line_num + 1
        # print line_num
        # print data[:5]
        return data


if __name__ == '__main__':
    # original_path = '/home/ubuntu-yk/github-workspace/medical_data/data-repository/'
    # Document2VecUtility.get_word_list(original_path + '2_0.txt', remove_html=True, stem_words=True)
    pass