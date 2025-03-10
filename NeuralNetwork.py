class NNode :
    def __init__(self, weights : list[float] = []):
        self.id = 0
        self.weights : list[float] = weights
        
class NNetwork :
    # Probably disimpan sebagai semacam list of lists, dimana satu NNetwork terdiri dari X lists dimana masing2 list
    # Ialah suatu layer tersendiri
    # Lalu satu node direpresentasikan dengan list of weights ke arah kanan?
    def __init__(self):
        self.layers : list[list[NNode]] = []
    
    def getLayerCount(self):
        return len(self.layers)
    
    def addLayer(self):
        self.layers.append([])
        
    def removeLayer(self,layer : int):
        try :
            self.layers.pop(layer)    
        except IndexError:
            print("Bruh illegal")
            
    def getLayer(self,layer : int):
        try :
            return self.layers[layer]    
        except IndexError:
            print("Bruh illegal")
            
    # Insert layer to node
    def addNode(self, layer : int, idx : int = -1, weights : list[float] = []):
        if (idx == -1) : #Set index to last if unsupplied
            idx = len(self.layers)
        self.layers[layer].insert(idx,NNode(weights=weights))
            
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
            finalResult = result
        return finalResult
        
            
    
    def calcTest(self, *args : float) -> list[float]:
        return self.calc(args)
    
    def __sanitizeWeightList(self, intendedLayer : int, weights : list):
        pass
        #if (len(weights))