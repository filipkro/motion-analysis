import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser


data_dirs = ('healthy-SLS', 'hipp-SLS', 'marked-SLS', 'musse-SLS',
             'shield-SLS', 'ttb-SLS')

action_poe = 'SLS_femval'

def main(args):
    corr = np.zeros(5)
    diff = np.zeros(5)
    amount = np.zeros(5)
    for dir in os.listdir(args.root):
        if dir in data_dirs:
            cohort = dir.split('-')[0]
            dir_path = os.path.join(args.root, dir)
            label_file = os.path.join(dir_path, cohort + '-labels.csv')
            labels = pd.read_csv(label_file, delimiter=',')

            combined = labels.filter(like=action_poe + '_mean').values
            rep_fields = [action_poe + str(i) for i in range(1,6)]
            reps = [labels.filter(like=field).values for field in rep_fields]
            reps = np.array(reps)[..., 0].T

            for row, score in zip(reps, combined):
                idx = np.where(row == score[0])[0]
                corr[idx] = corr[idx] + 1
                idx = ~np.isnan(row)
                amount[idx] = amount[idx] + 1
                diff[idx] = diff[idx] + np.abs(row[idx] - score[0])

    np.set_printoptions(precision=3)
    print(corr)
    print(diff)
    print(amount)
    weighted_diff = np.array([diff[i] / amount[i] for i in range(5)])
    print(weighted_diff)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    main(args)
