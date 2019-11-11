import numpy as np


def bind_function(input, smallest, largest):
    return (input - smallest) / (largest - smallest)

#to normalize between 0 and 1
def sigmoid(t):
    return 1/(1+np.exp(-t))

#to perform the delta weight updates by getting the derivatives
def sigmoid_derivative(p):
    return p * (1 - p)


class Neural:
    def __init__(self, x, y, learning_rate):
        self.input = x #initialize input
        self.bind_values(x)
        self.weights1 = [[0.2, 0.3, 0.2], [0.1, 0.1, 0.1]] #initialize output layer weights
        self.weights2 = [[0.5, 0.1]] #initialize hidden layer weights
        self.target = y #initialize the y variable
        self.bind_values(y)
        self.learning_rate = learning_rate #set the learning rate

	#to bind inputs to input vectors and output to output vectors	
    def bind_values(self, values):
        min_value = values[np.unravel_index(
            np.argmin(values, axis=None), values.shape)]
        max_value = values[np.unravel_index(
            np.argmax(values, axis=None), values.shape)]

        new_values = np.array([bind_function(i, min_value, max_value)
                               for i in np.nditer(values)], dtype=np.float64).reshape(values.shape)

        np.put(values, range(values.size), new_values)

    def feed_foward(self):
        new_inputs = []
        for epoch in self.input:
            epoch = np.array(epoch, ndmin=2).T
            layer1 = sigmoid(np.dot(self.weights1, epoch))
            layer2 = sigmoid(np.dot(self.weights2, layer1))
            new_inputs.append(layer2)

        return new_inputs

    def back_propagation(self, input, target):
        target_vector = np.array(target, ndmin=2).T
        input = np.array(input, ndmin=2).T

        output_vector1 = np.dot(self.weights1, input)
        output_vector_hidden = sigmoid(output_vector1)

        output_vector2 = np.dot(
            self.weights2, output_vector_hidden)
        output_vector_network = sigmoid(output_vector2) #activation

        output_errors = target_vector - output_vector_network # calculate loss

        print("Input: \n", input)
        print("Expected value: ", target)
        print("Actual value: ", output_vector_network)
        print("Error/Loss :", output_errors)

        
        tmp = output_errors * output_vector_network * \
            (1.0 - output_vector_network)

        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)

        self.weights2 += tmp

        
        hidden_errors = np.dot(self.weights2.T, output_errors)

        
        tmp = hidden_errors * output_vector_hidden * \
            (1.0 - output_vector_hidden)
        self.weights1 += self.learning_rate * \
            np.dot(tmp, input.T)

    def train(self):
        for i in range(len(self.input)):
            self.back_propagation(self.input[i], self.target[i])

    def print_matrices(self):
        print("First Layer: ", self.weights1)
        print("Second Layer: ", self.weights2)