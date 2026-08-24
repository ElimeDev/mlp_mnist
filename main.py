from mlp import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("data/mnist_train.csv")
test_data = pd.read_csv("data/mnist_test.csv")