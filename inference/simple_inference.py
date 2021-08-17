import sys
import os
import pickle

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

sys.path.append(os.path.join(BASE, 'classification/tsc/utils'))

from eval_vid import main as assess_subject
from argparse import ArgumentParser, ArgumentTypeError, Namespace


def main():
    f = open(os.path.join(BASE, 'inference/out/datasets.pkl'), 'rb')
    f100 = open(os.path.join(BASE, 'inference/out/datasets100.pkl'), 'rb')
    datasets = pickle.load(f)
    datasets100 = pickle.load(f100)
    f.close()
    f100.close()

    assess_subject(Namespace(), datasets=datasets, datasets100=datasets100, base_path=BASE)


if __name__ == '__main__':
    main()
