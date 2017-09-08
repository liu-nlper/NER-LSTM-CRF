# -*- coding: utf-8 -*-
"""
    使用gensim训练word vectors
    Usage:
        python3 train_word2vec_model.py path_of_corpus path_of_model
"""
import logging
import os.path
import sys
from gensim.models.word2vec import LineSentence
from time import time
from gensim.models import Word2Vec


if __name__ == '__main__':
    t0 = time()

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]  # corpus path and path to save model

    model = Word2Vec(
        sg=1, sentences=LineSentence(inp), size=50, window=5, min_count=3,
        workers=16, iter=40)

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.wv.save_word2vec_format(outp, binary=False)

    print('done in %ds!' % (time()-t0))
