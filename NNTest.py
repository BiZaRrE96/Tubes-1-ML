from NeuralNetwork import *

ntest = NNetwork(verbose=True)

ntest.addLayer()
ntest.addLayer()
ntest.addLayer()

ntest.addNode(0, weights=[0.15,0.25])
ntest.addNode(0, weights=[0.20,0.30])

ntest.addNode(1, weights=[0.40,0.50])
ntest.addNode(1, weights=[0.45,0.55])

ntest.addNode(2)
ntest.addNode(2)

ntest.bias = [[0.35,0.35],[0.6,0.6]]

print(ntest.calcTest(0.05,0.1))