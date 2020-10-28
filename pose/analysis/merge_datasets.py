import numpy as np
from argparse import ArgumentParser
from scipy.signal import find_peaks
# import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file1')
    parser.add_argument('np_file2')
    parser.add_argument('root')

    args = parser.parse_args()
    data1 = np.load(args.np_file1)
    data2 = np.load(args.np_file2)

    assert data1['mts'].shape[2] == data2['mts'].shape[2]

    min_len = np.min((data1['mts'].shape[1], data2['mts'].shape[1]))

    print(data1['mts'].shape)
    print(data2['mts'].shape)
    print(min_len)
    mts = None
    labels = None
    if data1['mts'].shape[1] == data2['mts'].shape[1]:
        mts = np.append(data1['mts'], data2['mts'], axis=0)
        labels = np.append(data1['labels'], data2['labels'])
    else:
        if data1['mts'].shape[1] == min_len:
            mts = data1['mts']
            labels = np.append(data1['labels'], data2['labels'])
            mts2 = data2['mts']
            crop = np.zeros((mts2.shape[0], min_len, mts2.shape[2]))
            mts_len = mts2.shape[1]
            for i in range(mts2.shape[0]):
                peak, _ = find_peaks(mts2[i, :, -1], distance=50,
                                     prominence=0.05, width=13)
                print(peak)
                print(i)
                # plt.plot(split_poses[i, :, -1])
                # plt.show()
                # if len(peak) > 0:
                print(len(peak))
                peak = peak[0]

                if peak <= int(min_len / 2):
                    crop[i, ...] = mts2[i, :min_len, ...]
                elif peak <= (mts_len - int(min_len / 2)):
                    print('IN IF')
                    print(int(np.ceil(min_len / 2)))
                    print(len(mts2[i, peak - int(np.ceil(min_len / 2)):
                                   peak - int(np.ceil(min_len / 2))
                                   + min_len, ...]))
                    crop[i, ...] = mts2[i, peak -
                                        int(np.ceil(min_len / 2)):
                                        peak -
                                        int(np.ceil(min_len / 2))
                                        + min_len, ...]
                else:
                    crop[i, ...] = mts2[i, -min_len:, ...]

        else:
            mts = data2['mts']
            labels = np.append(data2['labels'], data1['labels'])
            mts1 = data1['mts']
            crop = np.zeros((mts1.shape[0], min_len, mts1.shape[2]))
            mts_len = mts1.shape[1]
            for i in range(mts1.shape[0]):
                peak, _ = find_peaks(mts1[i, :, -1], distance=50,
                                     prominence=0.05, width=13)
                print(peak)
                print(i)
                # plt.plot(split_poses[i, :, -1])
                # plt.show()
                # if len(peak) > 0:
                print(len(peak))

                # plt.plot(mts1[i, :, -1])
                # plt.show()
                peak = peak[0]

                if peak <= int(min_len / 2):
                    crop[i, ...] = mts1[i, :min_len, ...]
                elif peak <= (mts_len - int(min_len / 2)):
                    print('IN IF')
                    print(int(np.ceil(min_len / 2)))
                    print(len(mts1[i, peak - int(np.ceil(min_len / 2)):
                                   peak - int(np.ceil(min_len / 2))
                                   + min_len, ...]))
                    crop[i, ...] = mts1[i, peak -
                                        int(np.ceil(min_len / 2)):
                                        peak -
                                        int(np.ceil(min_len / 2))
                                        + min_len, ...]
                else:
                    crop[i, ...] = mts1[i, -min_len:, ...]

        print(mts.shape)
        print(crop.shape)
        mts = np.append(mts, crop, axis=0)

    print(mts.shape)
    print(labels.shape)

    mts = mts[:, :, :-1]
    print(mts.shape)
    save_file = args.root + 'data.npz'
    np.savez(save_file, mts=mts, labels=labels)


if __name__ == '__main__':
    main()
