# See asserts in calc, this tries to prevents those from happening
# misalnya ada layer baru atau geser2, ini harus di handle juga sih nanti
def refresh(self):
    for layer in range(len(self.layers) - 1):
        for node in range(self.layers):
            while len(self.layers[layer][node]) < len(self.layers[layer+1]):
                self.layers[layer][node].append(0)

def calc(self, inputList : list[float]) -> list[float]:
    if len(inputList) != len(self.getLayer(0)):
        raise IndexError
    
    finalResult : list[float] = inputList
    
    for i in range(self.getLayerCount()-1): #itterate for every between-layers per-say
        
        # Constants
        currentLayerNodeCount = len(self.layers[i])
        nextLayerNodeCount = len(self.layers[i+1])
        
        result : list[float] = [0 for x in range(nextLayerNodeCount)] #result of every calc process
        
        assert (currentLayerNodeCount == len(finalResult)) #This gets "dragged" from left to right
        #Makes sure that there exists a corresponding node for every input
        
        for nodeIdx in range(currentLayerNodeCount) : #Itterate through every node
            node = self.layers[i][nodeIdx]
            assert len(node.weights) == nextLayerNodeCount
            
            #Simply we push calculate the results of that node
            for j in range(nextLayerNodeCount):
                result[j] += finalResult[nodeIdx] * node.weights[j]
            
        assert (len(result) == len(self.layers[i+1]))
        
        # Insert bias
        assert (nextLayerNodeCount == len(self.bias[i]))
        for j in range(nextLayerNodeCount):
            result[j] += self.bias[i][j]
        
        # EXPORT TO IMAGE OR SOMETHING FOR TRUE VALUES HERE OR SMTHN
        # activate
        if (self.verbose) :
            print(result,"==>",end=" ")
        self.activate(result,self.activation_array[i])
        
        finalResult = result
    return finalResult
    
def calcTest(self, *args : float) -> list[float]:
    return self.calc(args)

def __sanitizeWeightList(self, intendedLayer : int, weights : list):
    pass
    #if (len(weights))

def removeLayer(self,layer : int):
    try :
        self.layers.pop(layer)
        self.activation_array.pop(layer)
    except IndexError:
        print("Bruh illegal")
        
def getLayer(self,layer : int):
    try :
        return self.layers[layer]    
    except IndexError:
        print("Bruh illegal")
    
def popNode(self, layer : int, idx : int):
    l = self.getLayer(layer)
    if idx >= len(l):
        raise IndexError
    
    # Iterate for every layer before it if exists
    if (layer > 0 and layer < self.getLayerCount()):
        # cleanse layer-1
        for i in range(len(self.getLayer(layer-1))):
            #For every node in layer-1
            self.layers[layer-1][i].weights.pop(idx)
    
    self.layers[layer].pop(idx)

def activate(self,tempresult : list[float], method : str):
    if method == "sigmoid" :
        for i in range(len(tempresult)):
            tempresult[i] = (1 / (1 + exp(-tempresult[i])))
    elif method == "none":
        pass
    else :
        raise Exception("Undefined activation method")
    if (self.verbose) :
            print(tempresult)
    
def getLayerCount(self):
    return len(self.layers)

def print_weights(self):
    for i, layer in enumerate(self.layers[:-1]):
        print(f"Layer {i}:")
        for j, node in enumerate(layer):
            print(f"  Neuron {j}: Weights={node.weights}")
        print(f"  Bias: {self.bias[i]}")
    print()

def print_gradients(self, layers_to_print: list[int]):
    for layer_idx in layers_to_print:
        if layer_idx >= len(self.gradients) or layer_idx < 0:
            print(f"Layer {layer_idx} tidak valid.")
            continue
        
        print(f"Gradien Layer {layer_idx}:")
        for j, node_gradients in enumerate(self.gradients[layer_idx]):
            print(f"  Neuron {j}: Gradients={node_gradients}")
        print()