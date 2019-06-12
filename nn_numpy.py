import numpy as np 
import pickle
import sys
import cv2
from sklearn.datasets import load_digits 
from sklearn.datasets import load_iris 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.preprocessing import LabelBinarizer 
from sklearn.cross_validation import train_test_split

def tanh(x):  
    return np.tanh(x)

def tanh_deriv(x):  
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):  
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):  
    return logistic(x)*(1-logistic(x))



class NeuralNetwork:   
    def __init__(self, layers, activation='tanh'):  
        """  
        :param layers: A list containing the number of units in each layer.
        Should be at least two values  
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"  
        """  
        if activation == 'logistic':  
            self.activation = logistic  
            self.activation_deriv = logistic_derivative  
        elif activation == 'tanh':  
            self.activation = tanh  
            self.activation_deriv = tanh_deriv
    
        self.weights = []  
        for i in range(1, len(layers) - 1):  
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)  
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)
            
            
    def fit(self, X, y, learning_rate=0.2, epochs=1000, mini_batch = 0.3):         
        X = np.atleast_2d(X)         
        temp = np.ones([X.shape[0], X.shape[1]+1])         
        temp[:, 0:-1] = X  # adding the bias unit to the input layer         
        X = temp         
        y = np.array(y)
    
        for k in range(epochs): 
            cost = 0
            for i in range(0, X.shape[0]): 
                if np.random.randint(X.shape[0]) > X.shape[0] * mini_batch:
                    continue
                a = [X[i]]
        
                for l in range(len(self.weights)):  #going forward network, for each layer
                    a.append(self.activation(np.dot(a[l], self.weights[l])))  #Computer the node value for each layer (O_i) using activation function
                error = y[i] - a[-1]  #Computer the error at the top layer
                cost += np.linalg.norm(error)
                # if k > epochs - 2:
                #     print y[i], a[-1]
                deltas = [error * self.activation_deriv(a[-1])] #For output layer, Err calculation (delta is updated error)
                
                #Staring backprobagation
                for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer 
                    #Compute the updated error (i,e, deltas) for each node going from top layer to input layer 
                    deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
                deltas.reverse()  
                for i in range(len(self.weights)):  
                    layer = np.atleast_2d(a[i])  
                    delta = np.atleast_2d(deltas[i])  
                    self.weights[i] += learning_rate * layer.T.dot(delta)
            print cost, " in round %d" % k
                
                
    def predict(self, x):         
        x = np.array(x)         
        temp = np.ones(x.shape[0]+1)         
        temp[0:-1] = x         
        a = temp         
        for l in range(0, len(self.weights)):             
            a = self.activation(np.dot(a, self.weights[l]))         
        return a


if __name__ == "__main__":

    if sys.argv[1] == "train":
        digits = load_digits()
        # digits = load_iris()  
        X = digits.data  
        y = digits.target  
        X -= X.min() # normalize the values to bring them into the range 0-1  
        X /= X.max()

        X_train, X_test, y_train, y_test = train_test_split(X, y)  
        labels_train = LabelBinarizer().fit_transform(y_train)  
        labels_test = LabelBinarizer().fit_transform(y_test)
        nn = NeuralNetwork([len(X[0]),10,len(labels_train[0])],'logistic')  
        print "Start fitting..."
        nn.fit(X_train,labels_train,epochs=1000)  
        predictions = []  
        for i in range(X_train.shape[0]):  
            o = nn.predict(X_train[i] )  
            predictions.append(np.argmax(o))  
        # print confusion_matrix(y_test,predictions)  
        # print classification_report(y_test,predictions)
        print sum(1 if i == 0 else 0 for i in (predictions - y_train)) / (len(y_train)  * 0.01), "%% of %d samples" % len(y_train)
        with open('weights', 'wb') as modelfile:
            pickle.dump(nn, modelfile)
    elif sys.argv[1] == "test":
        with open('weights', 'rb') as modelfile:
            nn = pickle.load(modelfile)
        while True:
            path = raw_input("Input your digital image file directory:\n")
            img = cv2.imread(path)  
            img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
            b, g, r = cv2.split(img)
            b = [x / 255.0 for x in b.flatten()]
            # print b
            print np.argmax(nn.predict(b))
    else:
        print "Type right option!"