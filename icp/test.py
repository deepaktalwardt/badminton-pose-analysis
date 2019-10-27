import numpy as np
import time
import icp
import json
import matplotlib.pyplot as plt

# Constants
N = 100                                    # number of random points in the dataset
num_tests = 1                             # number of test iterations
dim = 2                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .3                            # max translation of the test set
rotation = .25                              # max rotation (radians) of the test set


def rotation_matrix(axis, th):
    axis = axis/np.sqrt(np.dot(axis, axis))

    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

def test_icp():
    total_time = 0

    A = np.random.rand(N, dim)

    for i in range(num_tests):
        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        np.random.shuffle(B)

        # Run ICP
        start = time.time()
        T, distances, iterations = icp.icp(B, A, tolerance=1e-10)
        total_time += time.time() - start

        plt.scatter(A[:,0], A[:,1], color='red')
        plt.scatter(B[:,0], B[:,1], color='blue')
        plt.show()

        # Make C a homogeneous representation of B
        C = np.ones((N, 3))
        C[:,0:2] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T

        plt.scatter(A[:,0], A[:,1], color='red')
        plt.scatter(B[:,0], B[:,1], color='blue')
        plt.scatter(C[:,0], C[:,1], color='green')
        plt.show()

        assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
        assert np.allclose(T[0:2,0:2].T, R, atol=6*noise_sigma)     # T and R should be inverses
        assert np.allclose(-T[0:2,2], t, atol=6*noise_sigma)        # T and t should be inverses

    print('icp time: {:.3}'.format(total_time/num_tests))

    return


if __name__ == "__main__":
    data = ''
    with open('data2.txt', "r") as read_file:
        data = json.load(read_file)
    
    POINTS = []

    for key in data:
        large_list = data[key]
        
        get_real_person = large_list[0]

        for item in large_list:
            if(len(item) > len(get_real_person)):
                get_real_person = item
        
        # iterate dictionary
        point = []
        for k, v in get_real_person.items():
            point.append(v)

        POINTS.append(point)

    T,d,i = icp.icp(np.asarray(POINTS[0]), np.asarray(POINTS[1]))
    print(T)
    print(d)
    print(i)