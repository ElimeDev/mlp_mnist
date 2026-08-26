from mlp import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("data/mnist_train.csv")
test_data = pd.read_csv("data/mnist_test.csv")

X_train = train_data.iloc[:, 1:].to_numpy() / 255
train_labels = train_data.iloc[:, 0].to_numpy()
y_train = np.eye(10)[train_labels]

X_test = test_data.iloc[:, 1:].to_numpy() / 255
test_labels = test_data.iloc[:, 0].to_numpy()
y_test = np.eye(10)[test_labels]

mlp = MLP([X_train.shape[1], 100, 10])
#mlp.set_hidden_layers_activation(relu, relu_prime)
mlp.set_cost_func(cross_entropy, cross_entropy_prime)

epochs = 60

mlp.train(X_train, y_train, epochs= epochs, learning_rate= 0.1, mini_batch_size= 10, lambda_= 5)

losses = mlp.get_last_training_data()["train_losses"]

pred = mlp.predict(X_test)
predicted_classes = np.argmax(pred, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print("accuracy : ", accuracy)

x = np.arange(1, len(losses) + 1)

plt.plot(x, losses, 'b-o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.grid(True)
plt.show()
