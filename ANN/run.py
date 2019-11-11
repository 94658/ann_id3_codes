import numpy as np

from Neuralcode import Neural

# Expected output given in the question
y = np.array([20, 15, 60, 70, 50, 40], dtype=np.float64)


# input given in the question
x = np.array([
    [30, 40, 50],
    [40, 50, 20],
    [50, 20, 15],
    [20, 15, 60],
    [15, 60, 70],
    [60, 70, 50]
], dtype=np.float64)




def main():
    size_of_learn_sample = int(len(x)*0.9)
    print(size_of_learn_sample)

    NN = Neural(x, y, 0.5)

    NN.train()
    NN.print_matrices()


if __name__ == "__main__":
    main()