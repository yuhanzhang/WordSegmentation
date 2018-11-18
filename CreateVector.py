import numpy as np
import gensim
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import GlobalParameter

class Word2vec:
    def __init__(self):
        self.word2vec = {}

    def load_word2vec(self, pre_word2vec, id2word, is_binary=False):
        """
        载入预训练的词向量，得到语料库中词语的词向量
        :param pre_word2vec: 预训练的词向量
        :param id2word:
        :param is_binary:
        :return: 语料库中词语的词向量
        """
        if not is_binary:
            f = open(pre_word2vec, errors='ignore')
            m, n = f.readline().split()
            word_size = int(n)

            for line in enumerate(f):
                line = line.strip('\n').strip().split()
                word = line[0]
                vector = [float(v) for v in line[1:]]
                if len(vector) != word_size:
                    print('word %d seems to be wrong' % word)
                    continue
                self.word2vec[word] = vector
        vocabulary_size = len(id2word)
        embedding = []
        bound = np.sqrt(6.0) / np.sqrt(vocabulary_size)
        word2vec_oov_count = 0

        for i in range(vocabulary_size):
            word = id2word[i]
            if word in self.word2vec:
                embedding.append(self.word2vec.get(word))
            else:
                # 随机赋值
                word2vec_oov_count += 1
                embedding.append(np.random.uniform(-bound, bound, word_size))

        print('word2vec oov count: %d' % word2vec_oov_count)
        return np.array(embedding)


def create_word2vec_corpus(train_file, valid_file, test_file, output_file):
    """
    将语料库全集处理成句子形式，输出到output_file中，作为训练word2vec的语料
    :param train_file:
    :param valid_file:
    :param test_file:
    :param output_file:
    :return:
    """
    sentence_count = 0
    f = open(output_file, 'w', encoding='utf-8')
    for filename in [train_file, valid_file, test_file]:
        if filename == '':
            continue
        print('processing word in file:', filename)
        with open(filename, 'r', encoding='utf-8') as fo:
            sentence = []
            for index, line in enumerate(fo):
                line = line.strip('\n')
                if not line:
                    f.write(' '.join(sentence) + '\n')
                    sentence.clear()
                    sentence_count += 1
                else:
                    sentence.append(line.split('\t')[0])
    f.close()
    print('There are %d sentences in all.' % sentence_count)


def pretrain_word2vec(corpus):
    """
    预训练词向量
    :param corpus:file path
    :return:
    """
    f = open(corpus, 'r', encoding='utf-8')
    sentences = f.readlines()
    all_word_list = []
    for sentence in sentences:
        all_word_list.append(sentence.split())
    print(len(all_word_list))
    model = gensim.models.Word2Vec(all_word_list, size=GlobalParameter.word_size, min_count=0, sg=1, hs=0, negative=6, iter=15, workers=64,
                                   window=5, seed=6)
    model.wv.save_word2vec_format('word2vec/word_embedding_300dim.txt', binary=False)