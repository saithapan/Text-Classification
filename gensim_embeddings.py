import gensim.downloader as api
import numpy as np

wv = api.load("glove-twitter-200")

def convert_txt_vctr(txt):
    vector_size = wv.vector_size
    wv_res = np.zeros(vector_size)
    ctr = 1
    for w in txt:
        if w in wv:
            ctr += 1
            wv_res += wv[w]
    wv_res = wv_res / ctr
    return wv_res
