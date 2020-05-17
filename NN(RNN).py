import numpy as np
import copy


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sig_der(output):
    return output * (1 - output)


# mapping integers to binary representation using binary decoding

int2binary = {}
# max length of binary numbers
binary_dim = 8

largest_num = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_num)], dtype=np.uint8).T, axis=1)
for i in range(largest_num):
    int2binary[i] = binary[i]

# training rate
step = 0.1
# number of inputs
input_size = 2
# size of hidden layer
hidden_size = 16
# output size
output_size = 1

# initializing weights
weight_0 = 2 * np.random.random((input_size, hidden_size)) - 1
weight_1 = 2 * np.random.random((hidden_size, output_size)) - 1
weight_h = 2 * np.random.random((hidden_size, hidden_size)) - 1

# updating weights
weight_0_update = np.zeros_like(weight_0)
weight_1_update = np.zeros_like(weight_1)
weight_h_update = np.zeros_like(weight_h)

# training network
Final_Error = 0
for i in range(100000):
    # generating simple addition problem a + b = c
    a_num = np.random.randint(largest_num / 2)
    a = int2binary[a_num]  # binary decoding

    b_num = np.random.randint(largest_num / 2)
    b = int2binary[b_num]  # binary decoding

    c_num = a_num + b_num
    c = int2binary[c_num]

    # where network stores its prediction
    d = np.zeros_like(c)

    # resetting error each iteration to see how each bit of data changes
    Error = 0

    layer_2_deltas = []
    layer1_values = []
    layer1_values.append(np.zeros(hidden_size))

    # moving through positions in binary decoding
    for pos in range(binary_dim):
        # generate input and output
        X = np.array([[a[binary_dim - pos - 1], b[binary_dim - pos - 1]]])
        y = np.array([[c[binary_dim - pos - 1]]]).T

        # hidden layer (input + previous)
        layer_1 = sigmoid(np.dot(X, weight_0) + np.dot(layer1_values[-1], weight_h))
        layer2 = sigmoid(np.dot(layer_1, weight_1))

        # calculating error
        layer2_error = y - layer2
        layer_2_deltas.append((layer2_error) * sig_der(layer2))
        Final_Error += np.abs(layer2_error[0])

        # decode estimates for print
        d[binary_dim - pos - 1] = np.round(layer2[0][0])

        # storing hidden layer for use later
        layer1_values.append(copy.deepcopy(layer_1))

    future_layer1_delta = np.zeros(hidden_size)

    for pos in range(binary_dim):
        X = np.array([[a[pos], b[pos]]])
        layer_1 = layer1_values[-pos - 1]
        prev_layer1 = layer1_values[-pos - 2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-pos-1]
        # error at hidden layer
        layer_1_delta = (future_layer1_delta.dot(weight_h.T)+layer_2_delta.dot(weight_1.T))*sig_der(layer_1)

        # updating weights
        weight_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        weight_h_update += np.atleast_2d(prev_layer1).T.dot(layer_1_delta)
        weight_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    weight_0 += weight_0_update * step
    weight_1 += weight_1_update * step
    weight_h += weight_h_update * step


    weight_0_update *= 0
    weight_1_update *= 0
    weight_h_update *= 0

    # print out progress
    if i % 1000 == 0:
        print("Error:" + str(Final_Error))
        print("Pred:" + str(d))
        print("True:" + str(c))

        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_num) + " + " + str(b_num) + " = " + str(out))

        print("------------")
