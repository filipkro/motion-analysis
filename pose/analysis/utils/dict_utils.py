import numpy as np


def generate_dict(name, poses, dim, data={}):
    dim_key = 'positions_' + dim
    subject = name[0:2]
    action = name.split('_')[0][2:]

    if subject not in data:
        data[subject] = {}

    if action not in data[subject]:
        data[subject][action] = {}

    data[subject][action][dim_key] = [poses.astype('float32')]

    return data


def main():
    first = {'01': {'SLS': {'positions_2d': np.array([[1, 2, 3], [4, 5, 6]])}}}
    sec = generate_dict('01FL1R_lol', np.array([[12, 13, 14], [1, 1, 1]]),
                        '3d', first)
    print(sec)
    third = generate_dict('01SLS_lol', np.array([[1, 2], [3, 4], [5, 6]]),
                          '3d', sec)
    print(third)
    fourth = generate_dict('02SLS_cool', np.array([[1], [2]]), '2d', third)
    print(fourth)
    print(generate_dict('01SLS_lol', np.array([[1, 2], [3, 4], [5, 6]]), '3d'))


if __name__ == '__main__':
    main()
