# Perceptron and Logistic Regression Code

---Overview---
This repository contains Python code implementing a Perceptron and Logistic Regression model for binary classification tasks. The code includes training on logical gates (AND, OR, NOT) and a dataset from a CSV file, with visualization of loss, accuracy, and decision boundaries.

---Features---
- **Perceptron Implementation**: Trains a simple perceptron on AND, OR, and NOT logic gates.
- **Logistic Regression**: Implements logistic regression with sigmoid activation, gradient descent, and loss calculation.
- **Data Handling**: Loads, shuffles, normalizes, and splits data into training and testing sets.
- **Visualization**: Plots loss over epochs, accuracy over epochs, and decision boundaries using matplotlib.
- **File I/O**: Saves and loads normalized training and testing data as NumPy files.

---Requirements---
- Python 3.x
- NumPy
- Matplotlib
- Scipy

---Installation---
1. Clone the repository or download the code.
2. Install required packages using pip:
3. Ensure a `data` directory exists with `test_data.csv` containing comma-separated values with features and labels.

---Usage---
1. Run the script directly:
2. The code will:
- Train the perceptron on AND, OR, and NOT gates.
- Load data from `test_data.csv`, normalize it, and split into training/testing sets.
- Train a logistic regression model and display results.
- Generate and save plots for loss, accuracy, and decision boundary in the `../Output/` directory.

## File Structure
- `Perceptron_Code.txt`: Main script containing all implementations.
- `data/`: Directory for input CSV and output NumPy files.
- `../Output/`: Directory for saving generated plots (e.g., `loss_plot.png`, `accuracy_plot.png`, `decision_boundary.png`).

## Functions
- `init_weights(n_features)`: Initializes random weights and bias.
- `step_function(x)`: Returns 1 if x > 0, else 0.
- `forward_pass(X, weight, bias)`: Computes perceptron output.
- `update_rule(weight, bias, X, y_i, y_hat_i, learning_rate)`: Updates weights and bias based on error.
- `train_perceptron(X, y, learning_rate, epochs)`: Trains the perceptron.
- `sigmoid_func(s)`: Applies the sigmoid function.
- `predict_probability(X, weight, bias)`: Computes probabilities using sigmoid.
- `get_predictions(X, weight, bias)`: Returns binary predictions.
- `calc_loss(y_true, y_predicted)`: Calculates the log loss.
- `gradient_func(X, y_true, y_predicted)`: Computes gradients for weight and bias updates.
- `update_func(weight, bias, delta_weight, delta_bias, learning_rate)`: Updates weights and bias using gradients.
- `model_train(X, y, learning_rate, epochs)`: Trains the logistic regression model.
- `plot_loss(loss_history)`, `plot_accuracy(accuracy_history)`, `plot_decision_boundary(x, y, weight, bias)`: Visualizes training metrics and decision boundary.

## Output
- Console output includes training progress, final weights, bias, mistakes per epoch, loss history, accuracy history, and test/train accuracies.
- Saved plots in `../Output/` directory visualize the training process and decision boundary.

## Notes
- Ensure the `data` directory is correctly set up with `test_data.csv`.
- The script assumes the last column of `test_data.csv` contains labels (0 or 1).
- Adjust `learning_rate` and `epochs` in the code for different training behaviors.
- The decision boundary plot labels have a minor error (both classes labeled as "Class 0"); update `label="Class 1"` for the green scatter plot if needed.

(For your reference, this is a simple implementation of a single-layer neural network)
---- Here the things that used in the code ----

Weight update rule:
  w = w + delta_w
  delta_w = learning_rate * (yi - y_hat_i) * xi
  learning_rate is between 0 and 1

Bias update rule:
  b = b + delta_b
  delta_b = learning_rate * (yi - y_hat_i)
  learning_rate is between 0 and 1

Approximation:
  y_hat_i = g(f(x)) = g(w * x + b)

---And Logic---

y   y_hat_i   |  Error (y - y_hat_i)   → Weight change
1     1       |         0              → No change
1     0       |         1              → Increase w, b
0     0       |         0              → No change
0     1       |        -1              → Decrease w, b

---Or Logic---

y   y_hat_i   |  Error (y - y_hat_i)   → Weight change
1     1       |         0              → No change
1     0       |         1              → Increase w, b
0     0       |         0              → No change
0     1       |        -1              → Decrease w, b

---Not Logic---

y   y_hat_i   |  Error (y - y_hat_i)   → Weight change
1     1       |         0              → No change
1     0       |         1              → Increase w, b
0     0       |         0              → No change
0     1       |        -1              → Decrease w, b
