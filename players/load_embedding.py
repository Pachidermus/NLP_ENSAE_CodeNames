from gensim.scripts.glove2word2vec import glove2word2vec
from datetime import datetime
import gensim

def load_w2v(path='/players/GoogleNews-vectors-negative300.bin') :
    """ load w2v model using given path (str) """ 
    start=datetime.now()
    model_w2v = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    print("Execution time :", datetime.now()-start)
    return model_w2v


def load_glove(folder='/players/', dim=300) :
    """ load GloVe model using given folder (str) and given dimension (int) """
    start=datetime.now()
    glove_input_file = folder+'glove.6B.'+str(dim)+'d.txt'
    word2vec_glove_file = folder+'glove.6B.'+str(dim)+'d.txt.word2vec'
    glove2word2vec(glove_input_file, word2vec_glove_file)

    model_glove = gensim.models.KeyedVectors.load_word2vec_format(word2vec_glove_file)
    print("Execution time :", datetime.now()-start)
    return model_glove