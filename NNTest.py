from NeuralNetwork import *

ntest = NNetwork("none")

ntest.addLayer()
ntest.addLayer()

ntest.addNode(0, weights=[0.5,1])
ntest.addNode(0, weights=[0.5,1])

ntest.bias = [[0,0]]

ntest.addNode(1)
ntest.addNode(1)

print(ntest.calcTest(4,4))