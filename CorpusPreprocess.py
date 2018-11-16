import re


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
            # print(line)
            sentences_index = re.finditer("。/w ”|？/w ”|！/w ”|。|？|！", line)
            # print(sentences)
            last_index = 0
            for sentence_index in sentences_index:
                # print(sentence_index)
                sentence = line[last_index:sentence_index.end()+2]
                # print(sentence)
                last_index = sentence_index.end() + 3
                # print('origin line: ', line)
                p = '\[(?P<phrase>[^/][^\[\]]+?)\]/\w+? '
                # p = '\[(?P<phrase>.+?)\]/\w+? '
                sentence = re.sub(p, self.__onlyPhrase, sentence)
                # print('new line: ', line)
                tokens = sentence.split()
                for token in tokens:
                    # word = ''.join(token.split('/')[:-1])
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
