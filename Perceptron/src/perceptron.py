import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from scipy.stats import alpha


# a = np.array([[1, 3], [2, 4]])
# b = np.array([[3, 4], [5, 6]])
# print(np.dot(a, b, out=None))

#---Linear Regression---#

def init_weights(n_features):
    weight = np.random.rand(n_features)
    bias = 0.0

    return weight, bias


def step_function(x):
    return np.where(x > 0, 1, 0)


def forward_pass(X, weight, bias):
    print("----Compute Perceptron Output: step_function(weight.input + bias)----\n")

    linear_output = np.dot(X, weight) + bias
    return step_function(linear_output)


# Testing before train the perception
# x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# weights, bias = init_weights(2)
# print("Output", forward_pass(x, weights, bias))

# X is the test data
# y_i is the desired output for the i-th sample input
# y_hat_i i the output of the perceptron for the i-th sample input
# learning rate is the step size for the weight update
def update_rule(weight, bias, X, y_i, y_hat_i, learning_rate):
    print("----Compute Weight and bias update----\n")

    error = (y_i - y_hat_i)
    weight += (learning_rate * error * X)
    bias += (learning_rate * error)
    return weight, bias


def train_perceptron(X, y, learning_rate, epochs):
    print("----Train the perceptron----\n")

    weights, bias = init_weights(X.shape[1])
    mistakes_per_epoch = []

    for epoch in range(epochs):
        mistakes_counter = 0
        for elem, label in zip(X, y):
            prediction = forward_pass(elem, weights, bias)
            if label != prediction:
                mistakes_counter += 1
                weights, bias = update_rule(weights, bias, elem, label, prediction, learning_rate)

        mistakes_per_epoch.append(mistakes_counter)

        if mistakes_counter == 0:
            print(f"Training completed after {epoch + 1} epochs with no mistakes.")
            break

    return weights, bias, mistakes_per_epoch


# Testing the perceptron training
# mpe - mistakes per epoch
# lr - learning rate

# Train for And Logic gate
print("----Train for And Logic gate----\n")
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # And gate output
lr = 0.1
epochs = 100

new_weights, new_bias, mpe = train_perceptron(x, y, lr, epochs)
print("Final Weights:", new_weights)
print("Final Bias:", new_bias)
print("Mistakes per epoch:", mpe)

predictions = forward_pass(x, new_weights, new_bias)
print("Desired output (AND Logic): ", predictions)
#
# #Train for Or Logic gate
print("----Train for Or Logic gate----\n")
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])  # Or gate output
lr = 0.1
epochs = 100

new_weights, new_bias, mpe = train_perceptron(x, y, lr, epochs)
print("Final Weights:", new_weights)
print("Final Bias:", new_bias)
print("Mistakes per epoch:", mpe)

predictions = forward_pass(x, new_weights, new_bias)
print("Desired output (OR Logic): ", predictions)
#
# #Train for Not Logic gate
print("----Train for Not Logic gate----\n")
x = np.array([[0], [1]])
y = np.array([1, 0])  # Not gate output
lr = 0.1
epochs = 100

new_weights, new_bias, mpe = train_perceptron(x, y, lr, epochs)
print("Final Weights:", new_weights)
print("Final Bias:", new_bias)
print("Mistakes per epoch:", mpe)

predictions = forward_pass(x, new_weights, new_bias)
print("Desired output (NOT Logic): ", predictions)

# Loading data from the file
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data")


def load_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    test_data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    col_1 = test_data[:, 0]
    col_2 = test_data[:, 1]

    # getting the last column as the label
    labels = test_data[:, -1]

    feature_arr = np.column_stack((col_1, col_2))

    # shuffling the data
    num_samples = feature_arr.shape[0]
    shuffled_indices = np.random.permutation(num_samples)
    feature_arr = feature_arr[shuffled_indices]
    labels = labels[shuffled_indices]

    return feature_arr, labels


# Splitting the data into training and testing sets
def split_data(features, labels):
    sample_no = features.shape[0]
    split_val = int(sample_no * 0.8)

    # training data
    x_train = features[:split_val]
    y_train = labels[:split_val]

    # testing data
    x_test = features[split_val:]
    y_test = labels[split_val:]

    return x_train, y_train, x_test, y_test


def normalize_data(x_train, x_test):
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)

    x_train_normalize = (x_train - x_train_mean) / x_train_std
    x_test_normalize = (x_test - x_train_mean) / x_train_std

    return x_train_normalize, x_test_normalize


features, labels = load_from_file(os.path.join(data_path, "test_data.csv"))
x_train, y_train, x_test, y_test = split_data(features, labels)
x_train_norm, x_test_norm = normalize_data(x_train, x_test)

#saving the data to numpy files
train_feature_path = os.path.join(data_path, "train_features.npy")
train_labels_path = os.path.join(data_path, "train_labels.npy")
test_feature_path = os.path.join(data_path, "test_features.npy")
test_labels_path = os.path.join(data_path, "test_labels.npy")

np.save(train_feature_path, x_train_norm)
np.save(train_labels_path, y_train)
np.save(test_feature_path, x_test_norm)
np.save(test_labels_path, y_test)

# Loading the saved data
train_features = np.load(train_feature_path)
train_labels = np.load(train_labels_path)
test_features = np.load(test_feature_path)
test_labels = np.load(test_labels_path)

# print("Train Features:", train_features)
# print("Train Labels:", train_labels)
# print("Test Features:", test_features)
# print("Test Labels:", test_labels)

#------ Logistic Regression ------#

#The Sigmoid function
def sigmoid_func(s):
    return 1 / (1 + np.exp(-s))

# The derivative of the Sigmoid function
def predict_probability(X, weight, bias):
    print("----Compute Sigmoid Output: sigmoid(weight.input + bias)----\n")

    linear_output = np.dot(X, weight) + bias
    return sigmoid_func(linear_output)

def get_predictions(X, weight, bias):
    print("----Compute Predictions----\n")

    probabilities = predict_probability(X, weight, bias)
    return np.where(probabilities >= 0.5, 1, 0)

#In this bigger the loss, worse the guss of the model
def calc_loss(y_true, y_predicted):
    print("----Compute Loss----\n")

    epsilon = 1e-15
    y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted))
    return loss

def gradient_func(X, y_true, y_predicted):
    print("----Compute Gradient----\n")

    no_sample = X.shape[0]
    error = y_predicted - y_true

    delta_weight = (1 /no_sample) * X.T.dot(error)
    delta_bias = (1 / no_sample) * np.sum(error)

    return delta_weight, delta_bias

#Test the gradient_func using a small example
# x = np.array([[1, 2], [2, 3], [3, 4]])
# y = np.array([0, 1, 0])
# p = np.array([0.2, 0.6, 0.3])
#
# print(gradient_func(x, y, p))

def update_func(weight, bias, delta_weight, delta_bias, learning_rate):
    print("----Compute Weights and Bias Update----\n")

    new_weight = (weight - learning_rate * delta_weight)
    new_bias = (bias - learning_rate * delta_bias)

    return new_weight, new_bias

# Example parameters
# w = np.array([0.5, -0.3])
# b = 0.1
#
# # Example gradients
# dw = np.array([0.1, 0.2])
# db = 0.05
#
# learning_rate = 0.01
#
# w, b = update_func(w, b, dw, db, learning_rate)
#
# print("Updated weights:", w)
# print("Updated bias:", b)


def model_train(X, y, learning_rate, epochs):
    print("----Train the Logistic Regression Model----\n")

    loss_history = []
    accuracy_history = []

    weight, bias = init_weights(X.shape[1])

    for epoch in range(epochs):
         y_predict = predict_probability(X, weight, bias)

         loss_count = calc_loss(y, y_predict)
         loss_history.append(loss_count)

         predict = (y_predict >=0.5).astype(int)
         accuracy_count = np.mean(predict == y)
         accuracy_history.append(accuracy_count)

         delta_weight, delta_bias = gradient_func(X, y, y_predict)
         weight, bias = update_func(weight, bias, delta_weight, delta_bias, learning_rate)

         if epoch % 10 == 0:
             print(f"Epoch {epoch}, Loss: {loss_count}, Accuracy: {accuracy_count}")

         if loss_count < 0.01:
             print(f"Training completed after {epoch + 1} epochs with loss: {loss_count}")
             break

    return weight, bias, loss_history, accuracy_history

#Train the model
weight, bias, loss_history, accuracy_history = model_train(train_features, train_labels, learning_rate=0.1, epochs=100)

y_predictions = predict_probability(test_features, weight, bias)
test_predictions = get_predictions(test_features, weight, bias)
train_predictions = get_predictions(train_features, weight, bias)

test_accuracy = np.mean(test_predictions == test_labels)
train_accuracy = np.mean(train_predictions == train_labels)
print("Test Accuracy: ", test_accuracy)
print("Train accuracy: ", train_accuracy)
print("Loss History: ", loss_history)
print("Accuracy History: ", accuracy_history)

def plot_loss(loss_history):

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Loss', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig("../Output/loss_plot.png")
    plt.show()

plot_loss(loss_history)

def plot_accuracy(accuracy_history):

    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_history, label='Accuracy', color='green')
    plt.title('Accuracy over Epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("../Output/accuracy_plot.png")
    plt.show()

plot_accuracy(accuracy_history)

def plot_decision_boundary(x, y, weight, bias):

    plt.figure(figsize=(8, 5))

    x_min = x[:, 0].min() - 1
    x_max = x[:, 0].max() + 1

    y_min = x[:, 1].min() - 1
    y_max = x[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    grd_points = np.c_[xx.ravel(), yy.ravel()]
    probs = predict_probability(grd_points, weight, bias)
    predictions = np.where(probs >= 0.5, 1, 0)
    predictions = predictions.reshape(xx.shape)

    plt.contour(xx, yy, predictions, alpha=0.4, cmap="RdYlBu")
    plt.scatter(x[y == 0, 0], x[y ==0, 1], color="red", label="Class 0")
    plt.scatter(x[y == 1, 0], x[y == 1, 1], color="green", label="Class 0")

    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.savefig("../Output/decision_boundary.png")
    plt.show()

plot_decision_boundary(train_features, train_labels, weight, bias)
