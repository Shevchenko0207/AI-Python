import numpy as np


class Perceptron:
    def __init__(self, input_size):
        """
        Initializes a simple perceptron model.

        Args:
            input_size (int): Number of input features.
        """
        # Ініціалізуємо ваги випадковими значеннями (наприклад, з нормального розподілу)
        # та зміщення (bias) нулем або невеликим випадковим числом
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def activation(self, x):
        """
        Activation function (Step function).

        Args:
            x (float): Input value.

        Returns:
            int: 1 if x >= 0, else 0.
        """
        return 1 if x >= 0 else 0

    def predict(self, x):
        """
        Predicts the output label using the perceptron model.

        Args:
            x (ndarray): Input features.

        Returns:
            int: Predicted label (1 or 0).
        """
        # Рахуємо скалярний добуток входів на ваги + додаємо зміщення
        linear_output = np.dot(x, self.weights) + self.bias
        # Пропускаємо через функцію активації
        result = self.activation(linear_output)
        return result

    def train(self, X, y, num_epochs, learning_rate):
        """
        Trains the perceptron model on the given dataset using the perceptron learning rule.

        Args:
            X (ndarray): Input features of the dataset.
            y (ndarray): Ground truth labels of the dataset.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for weight update.
        """
        for epoch in range(num_epochs):
            for i in range(X.shape[0]):
                # 1. Робимо прогноз для поточного прикладу
                prediction = self.predict(X[i])

                # 2. Рахуємо помилку (різницю між реальністю та прогнозом)
                error = y[i] - prediction

                # 3. Якщо є помилка, оновлюємо ваги та зміщення за правилом навчання перцептрону
                if error != 0:
                    self.weights += learning_rate * error * X[i]
                    self.bias += learning_rate * error
        return None


# XOR dataset: Input features and corresponding labels
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create and train the perceptron model
perceptron = Perceptron(input_size=2)
perceptron.train(X, y, num_epochs=1000, learning_rate=0.1)

# Test the trained model
print("--- Результати тестування одношарового перцептрону на XOR ---")
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    prediction = perceptron.predict(data)
    print(f"Input: {data}, Prediction: {prediction}")
