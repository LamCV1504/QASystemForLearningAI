import gzip
import gensim
import logging
from config import WINDOW, MIN_COUNT, CBOW, ALPHA, MAX_LEN
from config import get_path_currying

path_w2v_model = get_path_currying('Resources\\')

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def show_file_contents(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    """gzip format"""

    logging.info("Đọc file {0} ... ".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            if (i % 10 == 0):
                logging.info("đọc {0} ý kiến".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)


if __name__ == '__main__':

    # abspath = os.path.dirname(os.path.abspath(__file__))
    # data_file = os.path.join("D:\\Resources\\tt.txt.gz")
    data_file = path_w2v_model('tt.txt.gz') 
    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    logging.info("Đọc xong file dữ liệu")

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        vector_size=MAX_LEN,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=10,
        alpha=ALPHA,
        sg=CBOW, # CBOW
        cbow_mean=0)  # 0, use the sum of the context word vectors
    model.train(documents, total_examples=len(documents), epochs=20)
    # model.wv.save(os.path.join("D:\\Resources\\ALL_CHAPTER.model"))
    model.wv.save(path_w2v_model('ALL_CHAPTER.model'))



