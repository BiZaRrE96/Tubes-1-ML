from NeuralNetwork import *

ntest = NNetwork()

ntest.addLayer()
ntest.addLayer()

ntest.addNode(0, weights=[1,2])
ntest.addNode(0, weights=[1,2])

ntest.addNode(1)
ntest.addNode(1)

print(ntest.calcTest(4,4))