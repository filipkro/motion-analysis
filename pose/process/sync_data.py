import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def main():
    big_data = np.load('/home/filipkr/Documents/xjob/pose-data/big_data-centered.npz',
                       allow_pickle=True)['data'].item()

    subject = '02'
    action = 'SLS1R'

    for subject in big_data.keys():

        data3d = big_data[subject][action]['positions_3d']
        data2d = big_data[subject][action]['positions_2d'][0]

        print(subject)
        print(data3d.shape)
        print(data2d.shape)
        # print(data3d)
        # print(data2d)
        # plt.add_subplot
        # if data3d.shape[0] != data2d.shape[0]:
        if subject == '05':
            data2d = data2d[-data3d.shape[0]:, ...]
            print(data2d.shape)
        if subject == '18':
            data2d = data2d[-data3d.shape[0] + 15:, ...]
            data3d = data3d[:data2d.shape[0]:, ...]
            print(data2d.shape)
            print(data3d.shape)

        min_len = np.min((data3d.shape[0], data2d.shape[0]))
        data3d = data3d[:min_len, ...]
        data2d = data2d[:min_len, ...]
        print(data2d.shape)
        print(data3d.shape)
        plt.plot(data3d[:, 3, 0])
        plt.plot(data2d[:, 16, 0])

        plt.figure()
        plt.scatter(data3d[100, :, 0],
                    data3d[100, :, 1])

        plt.plot(data3d[100, [1, 0, 4], 0],
                 data3d[100, [1, 0, 4], 1])
        plt.plot(data3d[100, [1, 2, 3], 0],
                 data3d[100, [1, 2, 3], 1])
        plt.plot(data3d[100, [0, 6, 5], 0],
                 data3d[100, [0, 6, 5], 1])
        plt.plot(data3d[100, [8, 5, 7], 0],
                 data3d[100, [8, 5, 7], 1])
        # plt.tick_params(axis='x', labelsize=20)
        # plt.tick_params(axis='y', labelsize=20)
        plt.show()

        big_data[subject][action]['positions_2d'] = [data2d]
        big_data[subject][action]['positions_3d'] = [data3d]

    np.savez_compressed(
        '/home/filipkr/Documents/xjob/pose-data/big_data-synced',
        data=big_data)


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser
    main()
