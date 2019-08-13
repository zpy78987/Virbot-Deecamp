import pickle
import numpy as np

def read_fmy():    
    f = open('2019_0627_151718.fm_y','rb')
    fm_y = pickle.load(f)
    print(fm_y)
    print(np.array(fm_y).shape)

    f = open('2019_0627_151718-true.fm_y','rb')
    fm_y = pickle.load(f)
    print(fm_y)
    print(np.array(fm_y).shape)

    # f = open('2019_0627_151718.fm_y','wb')
    # fm_y = np.matrix(fm_y[0])
    # pickle.dump(fm_y, f)

def test_shuffle():
    test = [
        [
            [1, 2, 3],
            [1, 1]
        ],
        [
            [1, 2, 3, 4],
            [1,1,1]
        ],
        [
            [2],
            [3,3]
        ]

    ]
    print(test)                 # [[[1, 2, 3], [1, 1]], [[1, 2, 3, 4], [1, 1, 1]], [[2], [3, 3]]]
    np.random.shuffle(test)
    print(test)                 # [[[1, 2, 3], [1, 1]], [[2], [3, 3]], [[1, 2, 3, 4], [1, 1, 1]]]

def test_concate():
    test1 = [
        [
            [1, 2, 3],
            [1, 1, 1]
        ],
        [
            [2, 3, 4],
            [1, 1, 1]
        ]
    ]
    test2 = [
        [
            [1, 2, 3],
            [1, 1, 1]
        ],
        [
            [2, 3, 4],
            [1, 1, 1]
        ]
    ]
    print(np.array(test1).shape)
    resutl = np.concatenate((test1, test2), axis=2)
    print(resutl)

if __name__ == "__main__":
    read_fmy()