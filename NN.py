import numpy
import Utils as utils

def tanh(x):
    return (1.0 - numpy.exp(-2 * x)) / (1.0 + numpy.exp(-2 * x))


def tanh_derivative(x):
    return (1 + tanh(x)) * (1 - tanh(x))


class NeuralNetwork:
    # sieć składa się z listy liczb całkowitych, z podaniem
    # liczby neoronów w każdej warstwie
    def __init__(self, net_arch):
        #numpy.random.seed(0)
        self.activity = tanh
        self.activity_derivative = tanh_derivative
        self.layers = len(net_arch)
        self.steps_per_epoch = 10
        self.arch = net_arch

        self.weights = []
        # zakres wartości masy (-1,1)
        for layer in range(len(net_arch) - 1):
            w = 2 * numpy.random.rand(net_arch[layer] + 1, net_arch[layer + 1]) - 1
            self.weights.append(w)

    def fit(self, data, labels, learning_rate=0.1, epochs=10):
        # Dodanie przesunięć do warstwy wejścia
        ones = numpy.ones((1, data.shape[0]))
        Z = numpy.concatenate((ones.T, data), axis=1)
        training = epochs * self.steps_per_epoch

        for k in range(training):
            # if k % self.steps_per_epoch == 0:
            #     # print ('epochs:', k/self.steps_per_epoch)
            #     #print('epochs: {}'.format(k / self.steps_per_epoch))
            #     #for s in data:
            #        # print(s, nn.predict(s))

            sample = numpy.random.randint(data.shape[0])
            y = [Z[sample]]

            for i in range(len(self.weights) - 1):
                activation = numpy.dot(y[i], self.weights[i])
                activity = self.activity(activation)
                # dodaj przesunięcie do następnej warstwy
                activity = numpy.concatenate((numpy.ones(1), numpy.array(activity)))
                y.append(activity)

                # ostatnia warstwa
            activation = numpy.dot(y[-1], self.weights[-1])
            activity = self.activity(activation)
            y.append(activity)

            # błąd dla warstwy wyjścia
            error = labels[sample] - y[-1]
            delta_vec = [error * self.activity_derivative(y[-1])]

            # trzeba zacząć od tyłu — od przedostatniej warstwy
            for i in range(self.layers - 2, 0, -1):
                # delta_vec [1].dot(self.weights[i][1:].T)
                error = delta_vec[-1].dot(self.weights[i][1:].T)
                error = error * self.activity_derivative(y[i][1:])
                delta_vec.append(error)

            # odwrócenie
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            delta_vec.reverse()

            # propagacja wsteczna
            # 1. Pomnóż jego deltę wyjściową i aktywację wejścia
            #    aby uzyskać gradient wagi.
            # 2. Odejmij stosunek (procent) gradientu od wagi
            for i in range(len(self.weights)):
                layer = y[i].reshape(1, nn.arch[i] + 1)

                delta = delta_vec[i].reshape(1, nn.arch[i + 1])
                self.weights[i] += learning_rate * layer.T.dot(delta)

        print(self.weights)


    def predict(self, x):
        val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activity(numpy.dot(val, self.weights[i]))
            val = numpy.concatenate((numpy.ones(1).T, numpy.array(val)))

        return val


if __name__ == '__main__':

    nn = NeuralNetwork([9, 2, 4])

    data = utils.get_data("f.csv")
    labels = utils.get_labels("f.csv")
    data = numpy.array(data)
    labels = numpy.array(labels)

    l = numpy.zeros((3437, 4))

    class0 = [1, 0, 0, 0]
    class1 = [0, 1, 0, 0]
    class2 = [0, 0, 1, 0]
    class3 = [0, 0, 0, 1]

    for i, o in enumerate(labels):
        if labels[i] == 0:
            l[i][0] = class0[0]
            l[i][1] = class0[1]
            l[i][2] = class0[2]
            l[i][3] = class0[3]
        elif labels[i] == 1:
            l[i][0] = class1[0]
            l[i][1] = class1[1]
            l[i][2] = class1[2]
            l[i][3] = class1[3]
        elif labels[i] == 2:
            l[i][0] = class2[0]
            l[i][1] = class2[1]
            l[i][2] = class2[2]
            l[i][3] = class2[3]
        elif labels[i] == 3:
            l[i][0] = class3[0]
            l[i][1] = class3[1]
            l[i][2] = class3[2]
            l[i][3] = class3[3]

    nn.fit(data, labels, epochs=10)

    x = [1.01854751713818 / 10000, 24344 / 10000, 1.22625698324022 / 10000, 2.37011173184358 / 10000,
         2.09916201117318 / 10000, 2.96298882681564 / 10000, 2.6054469273743 / 10000, 3.34427374301676 / 10000, 2.83798882681564 / 10000]

    predict = nn.predict(x)
    print("Predict: {}".format(predict[1:]))


