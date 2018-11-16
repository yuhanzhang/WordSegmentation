import os
from CorpusPreprocess import Preprocessor


def gci(path):
    parents = os.listdir(path)
    for parent in parents:
        child = os.path.join(path, parent)
        if os.path.isdir(child):
            gci(child)
        else:
            if child[-1] != '_':
                cp = Preprocessor(child)
                cp.preprocess()

if __name__ == '__main__':
    # process corpus
    gci('E:\\chinese word segmentation\\corpus\\PeopleDaily2014')
    # cp = Preprocessor('c231190-20282163.txt')
    # cp.preprocess()