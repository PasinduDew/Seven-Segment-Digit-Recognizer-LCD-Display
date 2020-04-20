
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
import csv



# Neural Network Class Definition
class neuralNetwork:
    
    # Initialize the Nueral Network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        
        # Set number of nodes in each input, hidden, output layer
        self.inNodes = inputNodes
        self.hidNodes = hiddenNodes
        self.outNodes = outputNodes
        self.noOfLayers = 2
        
        # Learning rate
        self.learningRate = learningRate
        
        # -------------------------------- Initializing Weights --------------------------------------------------------
        """
            weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
            w11 w21
            w12 w22 etc
        """
        # -------------------- Simple Method ---------------------------------
        # self.wIH = (numpy.random.rand(self.hidNodes, self.inNodes) - 0.5)
        # self.wHO = (numpy.random.rand(self.outNodes, self.hidNodes) - 0.5)
        
        # -------------------- Advanced Method -------------------------------
        self.wIH = numpy.random.normal ( 0.0 , pow(self.inNodes, -0.5), (self.hidNodes, self.inNodes) )
        self.wHO = numpy.random.normal ( 0.0 , pow(self.hidNodes, -0.5), (self.outNodes, self.hidNodes) )
        
        # Activation Function is the Sigmoid Function
        self.activationFunction = lambda x: scipy.special.expit(x)

        pass
    
    
    
    # Train the Nueral Network
    def train(self, inputsList, targetsList):
        # Converts the Inputs List to a 2D Array
        inputs = numpy.array(inputsList, ndmin=2).T
        targets = numpy.array(targetsList, ndmin=2).T
        
        # Calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.wIH, inputs)
        # Calculate the signals emerging from hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)
        
        # Calculate signals into final output layer
        finalInputs = numpy.dot(self.wHO, hiddenOutputs)
        # Calculate the signals emerging from final output layer
        finalOutputs = self.activationFunction(finalInputs)
        
        # Output layer error is the (target actual)
        outputErrors = targets - finalOutputs
        
        # HIdden layer error is the output_errors, split by weights, recombined at hidden nodes
        hiddenErrors = numpy.dot(self.wHO.T, outputErrors)
        # Update the weights for the links between the hidden and output layers
        self.wHO += self.learningRate * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), numpy.transpose(hiddenOutputs))
        # Update the weights for the links between the input and hidden layers
        self.wIH += self.learningRate * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), numpy.transpose(inputs))
        pass
    
    # Query the Neural Network
    def query(self, inputsList):
        # convert inputs list to 2d array
        inputs = numpy.array(inputsList, ndmin=2).T
        
        # Calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.wIH, inputs)
        # Calculate the signals emerging from hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)
        
        # Calculate signals into final output layer
        finalInputs = numpy.dot(self.wHO, hiddenOutputs)
        # Calculate the signals emerging from final output layer
        finalOutputs = self.activationFunction(finalInputs)
        
        return finalOutputs
        pass

    # Get Weights Values
    def getWeights(self):
        return (self.wIH, self.wHO)

# --------------------------------------------------------------------------------------------------------------------
#             Saving the Weight Values of the Trained Neural Network -> to reduce the next execution time
# --------------------------------------------------------------------------------------------------------------------
    def saveWeights(self, outputFileNameCSV):
        
        rows = []

        for i in range(self.noOfLayers):
            weights = self.getWeights()[i]
            newRow = []
            newRow.append(weights.shape[0])
            newRow.append(weights.shape[1])

            for i in weights.flatten():
                newRow.append(i)

            rows.append(newRow)

        with open(outputFileNameCSV + ".csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)

# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
#                                          Load Weights From a CSV File
# --------------------------------------------------------------------------------------------------------------------
    def loadWeights(self, inputFileName):
        
        with open(inputFileName, 'r') as file:
            reader = csv.reader(file)
            line = 0
            for row in reader:
                if line == 0 :
                    self.wIH = numpy.asfarray(row[2:]).reshape(int(row[0]), int(row[1]))
                    line += 1
                elif line == 1 :
                    self.wHO = numpy.asfarray(row[2:]).reshape(int(row[0]), int(row[1]))
                    line += 1
                pass
            # print("== ", self.wIH)
            # print("Shape ", self.wIH.shape)



# --------------------------------------------------------------------------------------------------------------------


