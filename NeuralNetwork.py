import numpy


def tanh(x):
    return (1.0 - numpy.exp(-2*x)) / (1.0 + numpy.exp(-2*x))


def tanh_derivative(x):
    return (1 + tanh(x))*(1 - tanh(x))


class NeuralNetwork:

    def __init__(self, net_arch):
        self.activity = tanh
        self.activity_derivative = tanh_derivative
        self.layers = len(net_arch)
        self.steps_per_epoch = 1
        self.arch = net_arch
        self.weights = []
        for layer in range(self.layers - 1):
            w = 2*numpy.random.rand(net_arch[layer] + 1, net_arch[layer+1]) - 1
            self.weights.append(w)

    # def fit(self, data, labels, learning_rate=0.1, epochs=1):
    #     data = numpy.array(data)
    #     labels = numpy.array(labels)
    #     ones = numpy.ones((1, data.shape[0]))
    #     Z = numpy.concatenate((ones.T, data), axis=1)
    #     training = epochs*self.steps_per_epoch
    #     for k in range(training):
    #         if k % self.steps_per_epoch == 0:
    #             print('epochs: {}'.format(k/self.steps_per_epoch))
    #             for s in data:
    #                 print(s, self.predict(s))
    #                 sample = numpy.random.randint(data.shape[0])
    #                 y = [Z[sample]]
    #                 for i in range(len(self.weights)-1):
    #                     activation = numpy.dot(y[i], self.weights[i])
    #                     activity = self.activity(activation)
    #                     activity = numpy.concatenate((numpy.ones(1), numpy.array(activity)))
    #                     y.append(activity)
    #
    #                 #ostatnia warstwa
    #                 activation = numpy.dot(y[-1], self.weights[-1])
    #                 activity = self.activity(activation)
    #                 y.append(activity)
    #
    #                 #błąd dla wyjścia
    #                 error = labels[sample] - y[-1]
    #                 delta_vec = [error * self.activity_derivative(y[-1])]
    #                 for i in range(self.layers-2, 0, -1):
    #                     error = delta_vec[-1].dot(self.weights[i][1:].T)
    #                     error = error * self.activity_derivative(y[i][1:])
    #                     delta_vec.append(error)
    #                 delta_vec.reverse()
    #
    #                 for i in range(len(self.weights)):
    #                     layer = y[i].reshape(1, self.arch[i]+1)
    #                     delta = delta_vec[i].reshape(1, self.arch[i+1])
    #                     self.weights[i] += learning_rate * layer.T.dot(delta)
    #
    #     print(self.weights)

    def fit(self, data, labels, learning_rate=0.1, epochs=10):
        # Dodanie przesunięć do warstwy wejścia
        ones = numpy.ones((1, data.shape[0]))
        Z = numpy.concatenate((ones.T, data), axis=1)
        training = epochs * self.steps_per_epoch

        for k in range(training):
            if k % self.steps_per_epoch == 0:
                # print ('epochs:', k/self.steps_per_epoch)
                print('epochs: {}'.format(k / self.steps_per_epoch))
                for s in data:
                    print(s, nn.predict(s))

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

    def predict(self, x):
        val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activity(numpy.dot(val, self.weights[i]))
            val = numpy.concatenate((numpy.ones(1).T, numpy.array(val)))

        return val[1]

