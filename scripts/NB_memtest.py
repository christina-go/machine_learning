import functools as ft
import memory_profiler
from mlwpy import *


def nb_go(train_ftrs, test_ftrs, train_tgt):
    nb = naive_bayes.GaussianNB()
    fit = nb.fit(train_ftrs, train_tgt)
    preds = fit.predict(test_ftrs)


def split_data(dataset):
    split = skms.train_test_split(dataset.data, dataset.target, test_size=.25)
    return split[:-1]  #don't want test tgt


def msr_mem(go, args):
    base = memory_profiler.memory_usage()[0]
    mu = memory_profiler.memory_usage((go, args), max_usage=False)[0]
    print("{:<3}: ~{:.4f} MiB".format(go.__name__, mu - base))


if __name__ == "__main__":
    msr_mem(nb_go, split_data(datasets.load_iris()))