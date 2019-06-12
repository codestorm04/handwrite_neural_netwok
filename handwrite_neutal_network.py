# encoding:utf-8
# demo of MLP neural networks
import random
import sys
import numpy
import math
import pickle
import cv2
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

class nn:
    widths = list()
    depth = 3  # total depth
    w = list()
    bias = list()
    l = 0.2  # learning rate
    a = 0.2  # penalty term
    select_rate = 1.0  # statistical selecting rate
    precision_thr = 0.95  # the threshhold of the precision
    max_iteration = 50000  # the threshhold of the iteration
    converge_thr = 0.0001  # the threshhold of the w update

    def __init__(self, depth=3, widths=[4, 4, 3], l=0.1, a=0.1, select_rate=1.0):
        self.widths = widths
        self.depth = depth
        self.l = l
        self.a = a
        self.select_rate = select_rate

        for layer in range(0, self.depth - 1):
            w_layer = list()
            b_layer = list()
            for cell in range(0, self.widths[layer]):
                w_tmp = list()
                for cell_back in range(0, self.widths[layer + 1]):
                    w_tmp.append(random.random())
                    if cell == 0:  # bias exists except for input layer
                        b_layer.append(random.random())
                w_layer.append(w_tmp)
            self.w.append(w_layer)
            self.bias.append(b_layer)
            # w is a 3-D array, bias is 2-D

    # select to use GSD
    def __select(self, x, y):
        n_samgles = len(x)
        result_x = list()
        result_y = list()
        for i in range(0, 1): #int(n_samgles * self.select_rate)):
            sample = random.randint(0, n_samgles - 1)
            result_x.append(x[sample])
            result_y.append(y[sample])
        return result_x, result_y

    # calculate the outputs forward and return the outputs on each layers of each samples
    def __forward(self, x, y):
        # print x, y
        cost = 0
        outputs = list()
        correct_count = 0
        for sample in range(0, len(x)):  # samples
            outputs_x = list()
            outputs_x.append(x[sample])
            for layer in range(0, self.depth - 1):  # layers
                outputs_tmp = list()
                for back_cell in range(0, self.widths[layer + 1]):  # outputs
                    z = 0
                    for cell in range(0, self.widths[layer]):
                        z += self.w[layer][cell][back_cell] * outputs_x[layer][cell]
                    outputs_tmp.append(1 / (1 + math.exp(0 - z - self.bias[layer][back_cell])))
                outputs_x.append(outputs_tmp)
            outputs.append(outputs_x)

            # chech the precision and cost
            max_out = sys.maxsize * (-1)
            max_index = -1
            for cell in range(0, self.widths[-1]):
                if max_out < outputs_x[-1][cell]:
                    max_out = outputs_x[-1][cell]
                    max_index = cell
                cost += 0.5 * (((1 if cell == y[sample] else 0) - outputs_x[-1][cell]) ** 2)
            if max_index == y[sample]:
                correct_count += 1
        return outputs, correct_count / (len(outputs) * 1.0), cost

    def __backward(self, outputs, y):
        # bias_delta appended in w_delta
        errors = list()

        # compute the error backward
        for sample in range(0, len(outputs)):
            error_sample = list()
            for layer in range(self.depth - 1, 0, -1):
                error_layer = list()
                for cell in range(0, self.widths[layer]):
                    o = outputs[sample][layer][cell]
                    if layer == self.depth - 1:
                        error_tmp = ((1 if cell == y[sample] else 0) - o) * o * (1 - o)
                    else:
                        sigma = 0
                        for back_cell in range(0, self.widths[layer + 1]):
                            sigma += self.w[layer][cell][back_cell] * error_sample[0][back_cell]
                        error_tmp = sigma * o * (1 - o)
                    error_layer.append(error_tmp)
                error_sample.insert(0, error_layer)
            errors.append(error_sample)

        converged = True
        # update the w
        for layer in range(0, self.depth - 1):
            for cell in range(0, len(self.w[layer])):
                for back_cell in range(0, len(self.w[layer][cell])):
                    w_delta = list()
                    for sample in range(0, len(outputs)):
                        w_delta.append(self.l * outputs[sample][layer][cell] * errors[sample][layer][back_cell])
                    delta = numpy.mean(w_delta)
                    if converged and math.fabs(delta) > self.converge_thr:
                        converged = False
                    self.w[layer][cell][back_cell] += delta

            # update the bias
            for back_cell in range(0, self.widths[layer + 1]):
                b_tmp = list()
                for sample in range(0, len(outputs)):
                    b_tmp.append(self.l * errors[sample][layer][back_cell])
                self.bias[layer][back_cell] += numpy.mean(b_tmp)
        return converged

    def fit(self, x, y):
        count = 0
        precision = 0
        batch_size = 100
        correct_per_batch = 0
        cost = 0
        precisions = list()
        costs = list()

        while True:
            count += 1
            if count > self.max_iteration:
                print("Max_iteration reached!")
                break
            if count < self.max_iteration:
                x_select, y_select = self.__select(x, y)
            else:
                x_select = x
                y_select = y
            outputs, precision, cost = self.__forward(x_select, y_select)
            precisions.append(precision)
            costs.append(cost / 100.0)
            #print "%f of %d is correct, cost %f in %d round..." % (precision, len(x_select), cost, count),  outputs[0][-1]

            # for batch size > 1
            # if precision >= self.precision_thr or precision < 0.01:
            #     print "precision_threshhold reached!"
            #     break

            # for batch size = 1
            if precision >= 0.9:
                correct_per_batch += 1
            if count % batch_size == 0 and count >= batch_size:
                print("precision: %f in round %d at cost = %f" % (1.0 *correct_per_batch / (batch_size), count, cost))
                if correct_per_batch > batch_size * self.precision_thr:
                    print("precision threshold reached!")
                    break
                correct_per_batch = 0

            converged = self.__backward(outputs, y_select)
            if converged:
                print("network converged!")
                break

        print("Final procision %f in %d rounds!!! " % (precision, count))
        # plt.figure()
        # plt.plot(range(0, len(precisions)), precisions)
        # plt.plot(range(0, len(precisions)), costs)
        # plt.show()

    def predict(self, x):
        outputs, precision, cost = self.__forward(x, numpy.zeros(len(x)))
        print(numpy.argmax(outputs[0][-1]))


def get_data(sam_rate=0.7):
    iris = load_iris()
    # iris = load_digits()
    x_train = list()
    y_train = list()
    x_test = list()
    y_test = list()
    iris.data -= iris.data.min()
    iris.data /= iris.data.max()
    for i in range(0, len(iris.data)):
        if random.random() <= sam_rate:
            x_train.append(iris.data[i])
            y_train.append(iris.target[i])
        else:
            x_test.append(iris.data[i])
            y_test.append(iris.target[i])
    return x_train, y_train, x_test, y_test


def validate(nn, x, y):
    res = 0
    for i in range(0, len(x)):
        res += (1 if nn.predict(x[i]) == y[i] else 0)
    print("Final precision is: %f" % (res / float(len(x))))

def logistic(x):
    return 1/(1 + numpy.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

if __name__ == "__main__":
    if sys.argv[1] == "train":
        nn = nn()
        x_train, y_train, x_test, y_test = get_data(1.0)
        nn.fit(x_train, y_train)
        with open('nn.w', 'wb') as modelfile:
            pickle.dump(nn.w, modelfile)
        with open('nn.bias', 'wb') as modelfile:
            pickle.dump(nn.bias, modelfile)
        with open('nn.widths', 'wb') as modelfile:
            pickle.dump(nn.widths, modelfile)
        print("network model saved!")
        # validate(nn, x_test, x_test)
    elif sys.argv[1] == "test":
        nn = nn()
        with open('nn.w', 'rb') as modelfile:
            nn.w = pickle.load(modelfile)
        with open('nn.bias', 'rb') as modelfile:
            nn.bias = pickle.load(modelfile)
        with open('nn.widths', 'rb') as modelfile:
            nn.widths = pickle.load(modelfile)
        while True:
            path = input("Input your digital image file directory:\n")
            print("---------------" + path)
            img = cv2.imread(path)  
            img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
            b, g, r = cv2.split(img)
            b = [x / 255.0 for x in b.flatten()]
            # print b
            nn.predict([numpy.array(b)])
    else:
        print("Type right option!")

'''

Optimizations:
1. space and time complexity: 
        forward过程每层变量要记录；backward更新用旧的w；mini_batch要用1个更快否则平均后降低delta和收敛速度；
2. python matrix computing
4. penalty term
5. momentum (donglaign)
6. cross entropy
7. ReLU： 
        o * (1 - o) 因子不加收敛更快，类似ReLU解决梯度衰减，加了要增加迭代轮十倍以上；
8. structure parameters
9. cost precision converge:
        即使达到precision仍可能未converge或者cost不够小，应继续训练降低cost    
'''