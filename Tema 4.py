import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import numpy as np

MINI = -0.1
MAXI = 0.1
LEARNING_RATE = 0.5
EPOCHS = 5000
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
    options = [3, 4, 8, 6, 7]
    prob = [0, 0, 1, 0, 0]
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
    output_res = input
    # print("WE ARE AT INPUT", input)
    for i in range(0, HIDDEN_LAYERS + 1):
        input_res = np.dot(parameters["w" + str(i)], output_res) + parameters["b" + str(i)]
        input_clip = np.clip(input_res, -10000, 10000)
        #  print("input_rees ",input_res)
        output_res = sigmoid(input_clip)
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


# def squared_error(train_y, output):
#     errors = []
#     for i in range(len(train_y)):
#         error = np.sum((np.array(train_y[i]) - np.array(output[i])) ** 2)
#         errors.append(error)
#     return errors


def backward_propagation(input, real_output, output, layers, parameters):
    gradients = {key: arr.copy() for key, arr in parameters.items()}
    output_ar = np.array(output).reshape((len(output), 1))
    real_output_ar = np.array(real_output).reshape((len(real_output), 1))
    errors = output_ar * (1 - output_ar) * (output_ar - real_output_ar)
    errors = errors.tolist()
    errors = [item for sublist in errors for item in sublist]
    for i in range(len(gradients["w" + str(HIDDEN_LAYERS)])):
        for j in range(len(gradients["w" + str(HIDDEN_LAYERS)][i])):
            gradients["w" + str(HIDDEN_LAYERS)][i][j] = errors[i] * layers[HIDDEN_LAYERS][j]
    for i in range(len(gradients["w" + str(HIDDEN_LAYERS)])):
        gradients["b" + str(HIDDEN_LAYERS)][i] = errors[i]
    for i in range(HIDDEN_LAYERS - 1, -1, -1):
        new_errors = []
        for j in range(len(gradients["w" + str(i)])):
            add = 0
            for k in range(len(gradients["w" + str(i + 1)])):
                add += parameters["w" + str(i + 1)][k][j] * errors[k]
            val = layers[i+1][j] * (1 - layers[i+1][j]) * add
            new_errors.append(val)
        errors = new_errors
        for j in range(len(gradients["w" + str(i)])):
            # print(layers[i][j])
            for k in range(len(gradients["w" + str(i)][j])):
                gradients["w" + str(i)][j][k] = errors[j] * layers[i][k]
        for j in range(len(gradients["b" + str(i)])):
            gradients["b" + str(i)][j] = errors[j]
    return gradients


def train(train_set, real_labels, test_set, labels_test, parameters):
    global LEARNING_RATE

    mse_errors_train = []
    mse_errors_test = []
    epochs = []
    misclassified_train = []
    misclassified_test = []
    for epoch in range(EPOCHS):
        # if epoch == 2000:
        #     LEARNING_RATE = 0.3
        # elif epoch == 4000:
        #     LEARNING_RATE = 0.2
        error = mean_squared_error(real_labels, all_forward_propagation(train_set, parameters)[0])
        mse_errors_train.append(error)
        error = mean_squared_error(labels_test, all_forward_propagation(test_set, parameters)[0])
        mse_errors_test.append(error)
        epochs.append(epoch)
        print("Mean squared error - epoch", epoch, ": ", str(error * 100) + "%")
        all_gradients = {key: np.zeros_like(val) for key, val in parameters.items()}
        for p in range(len(train_set)):
            result, layers = forward_propagation(train_set[p], parameters)
            gradients = backward_propagation(train_set[p], real_labels[p], result, layers, parameters)
            for key, val in gradients.items():
                all_gradients[key] += val
        for key, val in all_gradients.items():
            all_gradients[key] = val / len(train_set)
        for j in range(HIDDEN_LAYERS + 1):
            parameters["w" + str(j)] -= all_gradients["w" + str(j)] * LEARNING_RATE
            parameters["b" + str(j)] -= all_gradients["b" + str(j)] * LEARNING_RATE

    out, _ = all_forward_propagation(train_set, parameters)
    for p in range(len(out)):
        out[p] = np.array(out[p])
        if np.argmax(out[p]) + 1 != np.argmax(real_labels[p]) + 1:
            misclassified_train.append((p, np.argmax(real_labels[p]) + 1))
    out, _ = all_forward_propagation(test_set, parameters)
    for p in range(len(out)):
        out[p] = np.array(out[p])
        if np.argmax(out[p]) + 1 != np.argmax(labels_test[p]) + 1:
            misclassified_test.append((p, np.argmax(labels_test[p]) + 1))
    get_class(train_set, real_labels, parameters)
    get_class(test_set, labels_test, parameters)
    tr_labels = []
    te_labels = []
    for k in range(len(real_labels)):
        tr_labels.append(real_labels[k].index(1) + 1)
    for k in range(len(labels_test)):
        te_labels.append(np.argmax(labels_test[k]) + 1)
    ou_tr = all_forward_propagation(train_set, parameters)[0]
    ou_te = all_forward_propagation(test_set, parameters)[0]
    for k in range(len(ou_tr)):
        ou_tr[k] = np.argmax(ou_tr[k]) + 1
    for k in range(len(ou_te)):
        ou_te[k] = np.argmax(ou_te[k]) + 1
    print("\033[33m---Train set---\033[0m")
    get_metrics(tr_labels, ou_tr)
    print("\033[33m---Test set---\033[0m")
    get_metrics(te_labels, ou_te)

    plt.plot(epochs, mse_errors_train, label='MSE Error Train')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Mean Squared Error vs Epochs')
    plt.legend()
    plt.savefig('mse_train.png')
    plt.show()

    plt.plot(epochs, mse_errors_test, label='MSE Error Test')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Mean Squared Error vs Epochs')
    plt.legend()
    plt.savefig('mse_test.png')
    plt.show()

    for idx, cls in misclassified_train:
        color = 'red' if cls == 1 else 'orange' if cls == 2 else 'yellow'
        plt.scatter(idx, cls, c=color)
    plt.xlabel('Data Point Index')
    plt.ylabel('Error')
    plt.title('Misclassified Points')
    plt.legend()
    plt.savefig('misclassified_train.png')
    plt.show()

    for idx, cls in misclassified_test:
        color = 'red' if cls == 1 else 'orange' if cls == 2 else 'yellow'
        plt.scatter(idx, cls, c=color, label=f'Misclassified Class {cls}')

    plt.xlabel('Data Point Index')
    plt.ylabel('Error')
    plt.title('Misclassified Points')
    plt.legend()
    plt.savefig('misclassified_test.png')
    plt.show()
    return parameters


def get_metrics(true_labels, predicted):
    accuracy = accuracy_score(true_labels, predicted)
    print(f'Accuracy: {accuracy * 100}%')
    precision = precision_score(true_labels, predicted, average='macro')
    recall = recall_score(true_labels, predicted, average='macro')
    f1 = f1_score(true_labels, predicted, average='macro')
    print(f'Precision: {precision * 100}%')
    print(f'Recall: {recall * 100}%')
    print(f'F1 Score: {f1 * 100}%')


def get_class(train_x, train_y, parameters):
    res = []
    output, _ = all_forward_propagation(train_x, parameters)
    correct = 0
    for i in range(len(output)):
        output[i] = np.array(output[i])
        # print(output[i])
        res.append(np.argmax(output[i]) + 1)
        if res[i] == np.argmax(train_y[i]) + 1:
            correct += 1
    print("\033[34mAccuracy: ", correct / len(output) * 100, "%\033[0m")
    print("\033[31mMean squared error:", mean_squared_error(train_y, output) * 100, "%\033[0m")


if __name__ == "__main__":
    input = input("Choose train(1) or see results(2): ")
    if input == "1":
        features, labels = load_dataset()
        train_x, train_y, test_x, test_y = divide_set(features, labels)
        parameters = initialize_parameters()
        new = transform_labels(train_y)
        min_vals = np.min(train_x, axis=0)
        max_vals = np.max(train_x, axis=0)
        train_x = (train_x - min_vals) / (max_vals - min_vals)
        min_vals = np.min(test_x, axis=0)
        max_vals = np.max(test_x, axis=0)
        test_x = (test_x - min_vals) / (max_vals - min_vals)
        lab = transform_labels(test_y)
        train(train_x, new, test_x, lab, parameters)
        with open("parameters.pkl", "wb") as fd:
            pickle.dump(parameters, fd)
        with open("test.pkl", "wb") as fd:
            pickle.dump(test_x, fd)
        with open("test_y.pkl", "wb") as fd:
            pickle.dump(test_y, fd)
    elif input == "2":
        with open("parameters.pkl", "rb") as fd:
            parameters = pickle.load(fd)
        with open("test.pkl", "rb") as fd:
            test_x = pickle.load(fd)
        with open("test_y.pkl", "rb") as fd:
            test_y = pickle.load(fd)
        min_vals = np.min(test_x, axis=0)
        max_vals = np.max(test_x, axis=0)
        test_x = (test_x - min_vals) / (max_vals - min_vals)
        lab = transform_labels(test_y)
        output, _ = all_forward_propagation(test_x, parameters)
        results = []
        for i in range(len(output)):
            results.append(np.argmax(output[i]) + 1)
        get_metrics(test_y, results)
