
RAW_CORPUS = 'E:\\chinese word segmentation\\corpus\\PeopleDaily2014'
RAW_DATA_DIR = 'raw_data'
WORD2VEC_DIR = 'word2vec'
WORD2VEC_FILE = WORD2VEC_DIR + '/word_embedding_300dim.txt'
TRAIN_FILE = 'data/train.txt'
VALID_FILE = 'data/valid.txt'
TEST_FILE = 'data/test.txt'
MAP_FILE = 'data/map.pkl'
SIZE_FILE = 'data/size.json'
MAX_LENGTH = 100

lstm_hidden_size = 300
word_size = 300
learning_rate = 0.001
input_dropout_keep = 1.0
dropout = 0.5
max_epoch = 20
batch_size = 32
optimizer = 'adm'
clip = 5
eval_step = 20