import math
import sys


def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = [line.strip().split() for line in lines]
        strings = [line[0] for line in data]
        return strings


def read_weights(filename):
    with open(filename, 'r') as file:
        weights = [float(weight) for weight in file.read().split()]
        print(weights)
        return weights


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def lucky_relu(x):
    return max(0.1 * x, x)



def predict_pattern(string, nn_weights):
    # Feed-forward neural network
    total_first_hidden = 0
    total_second_hidden = 0
    for i in range(len(string)):
        total_first_hidden += int(string[i]) * nn_weights[i]  # Input to first hidden layer computation

    hidden_first_activation = sigmoid(total_first_hidden)

    for i in range(16, 24):
        total_second_hidden += hidden_first_activation * nn_weights[i]  # Input to second hidden layer computation

    hidden_second_activation = lucky_relu(total_second_hidden)

    total_output = 0
    for i in range(24, 32):
        total_output += hidden_second_activation * nn_weights[i]  # Hidden layer to output computation

    output_threshold = (max(nn_weights[24:32]) + min(nn_weights[24:32])) / 2

    return 1 if total_output >= output_threshold else 0


def runnet(weights_file, data_file, output_file):
    nn_weights = read_weights(weights_file)
    data_strings = read_data(data_file)

    with open(output_file, 'w') as file:
        for string in data_strings:
            prediction = predict_pattern(string, nn_weights)
            file.write(string + " " + str(prediction) + '\n')


if __name__ == "__main__":
    weights_file = sys.argv[1]
    new_data_to_classify_file = sys.argv[2]
    runnet(weights_file, new_data_to_classify_file, 'output0.txt')
