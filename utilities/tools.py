import os
import numpy as np
import re


def w2v_format_to_numpy_array(in_file_addr, out_file_addr):
    import numpy as np

    with open(in_file_addr) as f:
        vocab_size, embeddings_size = map(int, f.readline().split())
        inp = f.readlines()

    out = np.zeros((vocab_size, embeddings_size))
    for i in range(vocab_size):
        nums_str = inp[i].split()[1:]
        out[i] = map(float, nums_str)

    np.save(out_file_addr, out)


def l2(vec):
    return np.sqrt(np.sum(vec ** 2))


def cosine_dist(x, y):
    return x.dot(y) / (x.dot(x) * y.dot(y) + 1E-3)


def keep_english(file_addr):
    with open(file_addr) as f:
        lines = [l for l in f.readlines() if l != '']
    lines = map(lambda x: re.sub(r'[^\w]', ' ', x), lines)  # removes non-english characters
    lines = map(lambda x: re.sub(r'\s+', ' ', x), lines)
    with open(file_addr, 'w') as f:
        f.writelines(lines)


def keep_english_dir(dir_name):
    for doc in os.listdir(dir_name):
        keep_english(os.path.join(dir_name, doc))
