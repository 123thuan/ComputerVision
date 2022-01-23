import numpy as np


class NeuralNetwork():

    def __init__(self):
        # tạo số ngẫu nhiên
        np.random.seed(1)

        # đảo ngược trọng số thành ma trận 3 x 1 với các giá trị từ -1 đến 1
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        # applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # tính đạo hàm cho hàm sigmoid
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        # đào tạo mô hình để đưa ra dự đoán chính xác trong khi điều chỉnh trọng số liên tục
        for iteration in range(training_iterations):
            # đưa dữa liệu dào tạo qua no ron
            output = self.think(training_inputs)

            # tỷ lệ lỗi tính toán cho lan truyền ngược
            error = training_outputs - output

            # thực hiện điều chỉnh trọng số
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        # chuyển các đầu vào thông qua nơ-ron để nhận được đầu ra
        # chuyển đổi giá trị thành số thực

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":
    # initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    # training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

    training_outputs = np.array([[0, 1, 1, 0]]).T #hàm T chuyển ngang sang doc

    # training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    user_input_one = str(input("Input One: "))
    user_input_two = str(input("Input Two: "))
    user_input_three = str(input("Input Three: "))

    print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
    print("New Output data: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))