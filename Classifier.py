from NeuralNetwork import NeuralNetwork
import Utils as utils

if __name__ == '__main__':
    nn = NeuralNetwork([9, 4, 4])

    data = utils.get_data("f.csv")
    labels = utils.get_labels("f.csv")



    #nn.fit(data, labels)

    x = [1.01854751713818, 24344, 1.22625698324022, 2.37011173184358,
         2.09916201117318, 2.96298882681564, 2.6054469273743, 3.34427374301676, 2.83798882681564]

    #print("Predict: {}".format(nn.predict(x)))

