import numpy as np

MINI = -0.1
MAXI = 0.1
LEARNING_RATE = 0.01
EPOCHS = 500
INPUT_SIZE = 7
OUTPUT_SIZE = 3
HIDDEN_LAYERS = 1


def load_dataset():
    features = []
    labels = []
    index = 0
    with open("seeds_dataset.txt", "r") as fd:
        for line in fd:
            if line == "\n":
                continue
            line = line.strip()
            line = line.split()
            features.append([])
            for i in range(len(line) - 1):
                try:
                    value = float(line[i])
                except ValueError:
                    print("Value error")
                    exit(1)
                features[index].append(value)
            labels.append(int(line[-1]))
            index += 1
    return features, labels


def divide_set(feat, lab, partition=0.8):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    shuffle = np.random.permutation(len(feat))
    feat = np.array(feat)[shuffle]
    lab = np.array(lab)[shuffle]
    training_size = int(partition * len(feat))
    for i in range(len(feat)):
        if i < training_size:
            train_x.append(feat[i])
            train_y.append(lab[i])
        else:
            test_x.append(feat[i])
            test_y.append(lab[i])
    return train_x, train_y, test_x, test_y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def mean_squared_error(true, predicted):
    n = len(true)
    error = 0
    for i in range(n):
        true_i = np.array(true[i])
        predicted_i = np.array(predicted[i])
        error += np.sum((true_i - predicted_i) ** 2)
    error = error / (2 * n)
    return error


def transform_labels(labels):
    new_labels = []

    for i in range(len(labels)):
        if labels[i] == 1:
            new_labels.append([1, 0, 0])
        elif labels[i] == 2:
            new_labels.append([0, 1, 0])
        else:
            new_labels.append([0, 0, 1])
    return new_labels


def initialize_parameters():
    options = [3, 4, 5, 6, 7]
    prob = [0.1, 0.15, 0.5, 0.15, 0.1]
    param = {}
    prec = INPUT_SIZE
    for i in range(0, HIDDEN_LAYERS):
        number_of_neurons = np.random.choice(options, p=prob)
        mat = np.random.uniform(MINI, MAXI, (number_of_neurons, prec))
        param["w" + str(i)] = mat
        param["b" + str(i)] = np.random.uniform(MINI, MAXI, (number_of_neurons, 1))
        prec = number_of_neurons
    mat = np.random.uniform(MINI, MAXI, (OUTPUT_SIZE, prec))
    param["w" + str(HIDDEN_LAYERS)] = mat
    param["b" + str(HIDDEN_LAYERS)] = np.random.uniform(MINI, MAXI, (OUTPUT_SIZE, 1))
    return param


def forward_propagation(input, parameters):
    input = np.array(input).reshape((len(input), 1))
    layers = []
    inputs_list = input.tolist()
    good_list = [item for sublist in inputs_list for item in sublist]
    layers.append(good_list)
    # input_res = np.dot(parameters["w0"], input) + parameters["b0"]
    # output_res = sigmoid(input_res)
    # outputs_list = output_res.tolist()
    # good_list = [item for sublist in outputs_list for item in sublist]
    output_res = input
    for i in range(0, HIDDEN_LAYERS + 1):
        input_res = np.dot(parameters["w" + str(i)], output_res) + parameters["b" + str(i)]
        output_res = sigmoid(input_res)
        outputs_list = output_res.tolist()
        good_list = [item for sublist in outputs_list for item in sublist]
        layers.append(good_list)
    return good_list, layers


def all_forward_propagation(train, parameters):
    output = []
    layers = []
    for i in range(len(train)):
        output.append(forward_propagation(train[i], parameters)[0])
        layers.append(forward_propagation(train[i], parameters)[1])
    return output, layers


def get_values(layers, index):
    values = []
    for i in range(len(layers)):
        values.append(layers[i][index])
    return values


def squared_error(train_y, output):
    errors = []
    for i in range(len(train_y)):
        error = np.sum((np.array(train_y[i]) - np.array(output[i])) ** 2)
        errors.append(error)
    return errors


def backward_propagation(train_x, train_y, output, layers, parameters):
    # error = mean_squared_error(train_y, output)
    error = np.array(train_y) - np.array(output)
    # error = squared_error(train_y, output)
    # print(error)
    output = np.array(output)
    delta_output = sigmoid_derivative(output) * error
    # print(delta_output)
    layer_values = get_values(layers, HIDDEN_LAYERS)
    el_ar = np.array(layer_values)
    parameters["w" + str(HIDDEN_LAYERS)] += LEARNING_RATE * np.dot(delta_output.T, el_ar)
    parameters["b" + str(HIDDEN_LAYERS)] += LEARNING_RATE * np.mean(delta_output, axis=0).reshape((3, 1))
    delta_hidden = delta_output
    for i in range(HIDDEN_LAYERS - 1, 0, -1):
        layer_values = get_values(layers, i)
        delta_hidden = sigmoid_derivative(layer_values) * np.dot(delta_hidden, parameters["w" + str(i)].T)
        parameters["w" + str(i)] += LEARNING_RATE * np.dot(delta_hidden.T, np.array(get_values(layers, i - 1)))
        parameters["b" + str(i)] += LEARNING_RATE * np.sum(delta_hidden, axis=0).reshape((len(layer_values[0]), 1))
    layer_values = get_values(layers, 1)
    delta_hidden = sigmoid_derivative(np.array(layer_values)) * np.dot(delta_hidden, parameters["w1"])
    parameters["w0"] += LEARNING_RATE * np.dot(delta_hidden.T, np.array(get_values(layers, 0)))
    parameters["b0"] += LEARNING_RATE * np.mean(delta_hidden, axis=0).reshape((len(layer_values[0]), 1))

    return parameters


def get_class(train_x, train_y, parameters):
    res = []
    output, _ = all_forward_propagation(train_x, parameters)
    # print(output)
    for i in range(len(output)):
        output[i] = np.array(output[i])
        # print(output[i])
        res.append(np.argmax(output[i]) + 1)
        print(res[i], train_y[i])


if __name__ == "__main__":
    features, labels = load_dataset()
    train_x, train_y, test_x, test_y = divide_set(features, labels)
    parameters = initialize_parameters()
    # output = all_forward_propagation(train_x, parameters)
    # print(output[0])
    #  print("qwsdsf", output[1][0])
    output = all_forward_propagation(train_x, parameters)
    print(output[0])
    output2 = forward_propagation(train_x[0], parameters)
    print("Result: ", output2[0])
    layer1 = output2[1][0]
    layer2 = output2[1][1]
    layer3 = output2[1][2]
    print("First layer: ", layer1)
    print("Second layer: ", layer2)
    print("Last layer: ", layer3)
    # new = transform_labels(train_y)
    # for epoch in range(EPOCHS):
    #     output = all_forward_propagation(train_x, parameters)
    #     parameters = backward_propagation(train_x, new, output[0], output[1], parameters)
    # get_class(train_x, train_y, parameters)
# print(new[0])
# print(train_y[0])
# print(output)
