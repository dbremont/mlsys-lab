import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

dataset = [
    ([0, 0], 0),
    ([1, 0], 0),
    ([0, 1], 0),
    ([1, 1], 1)
]

def rand():
    return random.uniform(-1, 1)

# Layer 1 weights (2x2)
w11 = rand()
w12 = rand()
w21 = rand()
w22 = rand()

# Layer 1 biases
b1 = rand()
b2 = rand()

# Layer 2 weights
v1 = rand()
v2 = rand()

# Output bias
b3 = rand()

learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):

    total_loss = 0

    for inputs, y in dataset:

        x1 = inputs[0]
        x2 = inputs[1]

        # -------------------------
        # FORWARD PASS
        # -------------------------

        # Hidden neuron 1
        z1 = w11 * x1 + w12 * x2 + b1
        a1 = sigmoid(z1)

        # Hidden neuron 2
        z2 = w21 * x1 + w22 * x2 + b2
        a2 = sigmoid(z2)

        # Output neuron
        z3 = v1 * a1 + v2 * a2 + b3
        y_hat = sigmoid(z3)

        # -------------------------
        # LOSS
        # -------------------------

        loss = -(y * math.log(y_hat + 1e-9) +
                (1 - y) * math.log(1 - y_hat + 1e-9))

        total_loss += loss

        # -------------------------
        # BACKPROPAGATION
        # -------------------------

        # Output error
        delta3 = y_hat - y

        # Gradients output weights
        dv1 = delta3 * a1
        dv2 = delta3 * a2
        db3 = delta3

        # Hidden layer errors
        delta1 = (v1 * delta3) * sigmoid_derivative(z1)
        delta2 = (v2 * delta3) * sigmoid_derivative(z2)

        # Gradients hidden weights
        dw11 = delta1 * x1
        dw12 = delta1 * x2

        dw21 = delta2 * x1
        dw22 = delta2 * x2

        db1 = delta1
        db2 = delta2

        # -------------------------
        # GRADIENT DESCENT UPDATE
        # -------------------------

        v1 -= learning_rate * dv1
        v2 -= learning_rate * dv2
        b3 -= learning_rate * db3

        w11 -= learning_rate * dw11
        w12 -= learning_rate * dw12
        w21 -= learning_rate * dw21
        w22 -= learning_rate * dw22

        b1 -= learning_rate * db1
        b2 -= learning_rate * db2

    if epoch % 1000 == 0:
        print("epoch:", epoch, "loss:", total_loss)

print("\nTesting network\n")

for inputs, y in dataset:

    x1 = inputs[0]
    x2 = inputs[1]

    z1 = w11 * x1 + w12 * x2 + b1
    a1 = sigmoid(z1)

    z2 = w21 * x1 + w22 * x2 + b2
    a2 = sigmoid(z2)

    z3 = v1 * a1 + v2 * a2 + b3
    y_hat = sigmoid(z3)

    print(inputs, "pred:", round(y_hat,3), "target:", y)