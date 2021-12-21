from encodingsource import InitializerUniformlyRotation
from sf import sfGenerator
from hsgs import hsgsGenerator
import numpy as np
import math
import random
from qiskit import execute, Aer, QuantumRegister, QuantumCircuit, ClassicalRegister
from sympy.combinatorics.graycode import GrayCode
from qiskit.aqua.utils.controlled_circuit import apply_cu3
from extrafunctions import *
from neuron import *
np.random.seed(7)

print("--setosa--")

df = pd.read_csv('setosa.csv')
x_train = df.iloc[:, :-1].to_numpy()
y_train = df.iloc[:, -1].to_numpy()

data_len = len(y_train)
input_dim = x_train.shape[-1]
nb_epochs = 20
lr = 0.01
threshold = 0.001
simulator = Aer.get_backend('qasm_simulator')

trainNeuron(method='delta',
            nb_epochs=1000,
            binaryWeights=True,
            stochastic=False,
            listOfInput=x_train,
            listOfExpectedOutput=y_train,
            circuitGeneratorOfUOperator='encoding-input',
            simulator=simulator,
            threshold=threshold,
            lr = lr,
            memoryOfExecutions={})

# print("--virginica--")

# df = pd.read_csv('virginica.csv')
# x_train = df.iloc[:, :-1].to_numpy()
# y_train = df.iloc[:, -1].to_numpy()

# data_len = len(y_train)
# input_dim = x_train.shape[-1]
# nb_epochs = 20
# lr = 0.01
# threshold = 0.001
# simulator = Aer.get_backend('qasm_simulator')

# trainNeuron(method='delta',
#             nb_epochs=100,
#             binaryWeights=True,
#             stochastic=False,
#             listOfInput=x_train,
#             listOfExpectedOutput=y_train,
#             circuitGeneratorOfUOperator='encoding-input',
#             simulator=simulator,
#             threshold=threshold,
#             lr = lr,
#             memoryOfExecutions={})            

# print("--versicolor--")

# df = pd.read_csv('versicolor.csv')
# x_train = df.iloc[:, :-1].to_numpy()
# y_train = df.iloc[:, -1].to_numpy()

# data_len = len(y_train)
# input_dim = x_train.shape[-1]
# nb_epochs = 20
# lr = 0.01
# threshold = 0.001
# simulator = Aer.get_backend('qasm_simulator')

# trainNeuron(method='delta',
#             nb_epochs=100,
#             binaryWeights=True,
#             stochastic=False,
#             listOfInput=x_train,
#             listOfExpectedOutput=y_train,
#             circuitGeneratorOfUOperator='encoding-input',
#             simulator=simulator,
#             threshold=threshold,
#             lr = lr,
#             memoryOfExecutions={})                

