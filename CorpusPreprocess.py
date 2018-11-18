import re
import os


class Preprocessor:
    """语料预处理"""
    def __init__(self, filename):
        self.filename = filename

    def __onlyPhrase(self, matched):
        return matched.group('phrase')+' '

    def preprocess(self):
        file = open(self.filename, 'r', encoding='utf-8')
        line = file.readline()
        new_content = []
        while line:
            sentences_index = re.finditer("。/w ”|？/w ”|！/w ”|。|？|！", line)
            last_index = 0
            for sentence_index in sentences_index:
                sentence = line[last_index:sentence_index.end()+2]
                last_index = sentence_index.end() + 3                # print('origin line: ', line)
                p = '\[(?P<phrase>[^/][^\[\]]+?)\]/\w+? '
                # 将匹配的"[...]/.."替换成[]内的内容
                sentence = re.sub(p, self.__onlyPhrase, sentence)
                tokens = sentence.split()
                for token in tokens:
                    word = token[:token.rfind('/')]
                    if len(word) == 1:
                        new_content.append(word+'\t'+'S')
                    else:
                        try:
                            new_content.append(word[0]+'\t'+'B')

                            for i, character in enumerate(word[1:-1]):
                                new_content.append(character+'\t'+'M')
                            new_content.append(word[-1]+'\t'+'E')
                        except IndexError:
                            print(self.filename)
                new_content.append('')
            line = file.readline()
        new_filename = 'data' + self.filename[self.filename.rfind('\\'):]
        newfile = open(new_filename, 'w', encoding='utf-8')
        newfile.write('\n'.join(new_content))


def gci(path):
    """
    递归遍历文件夹中的所有文件
    :param path: path of the folder
    :return:
    """
    parents = os.listdir(path)
    for parent in parents:
        child = os.path.join(path, parent)
        if os.path.isdir(child):
            gci(child)
        else:
            if child[-1] != '_':
                cp = Preprocessor(child)
                cp.preprocess()


def create_data(rawdata_dir):
    """
    将语料分割成train, valid, test
    :param rawdata_dir:原始语料文件夹路径
    :return:
    """
    parents = os.listdir(rawdata_dir)
    all_data = []
    for parent in parents:
        child = os.path.join(rawdata_dir, parent)
        f = open(child, 'r', encoding='utf-8')
        all_data += f.readlines()
        f.close()

    sentence_num = 0
    for data in all_data:
        if data == '\n':
            sentence_num += 1

    print('the number of sentences is :', sentence_num)

    valid_num = int(sentence_num / 50)
    print('the number of valid sentences is :', valid_num)
    valid_data = []
    count = 0
    total = 0
    for data in all_data:
        if count == valid_num:
            break
        valid_data.append(data)
        total += 1
        if data == '\n':
            count += 1
    print(total)
    f = open('data/valid.txt', 'w', encoding='utf-8')
    f.writelines(valid_data)
    f.close()
    f = open('data/train.txt', 'w', encoding='utf-8')
    f.writelines(all_data[total:])
    f.close()