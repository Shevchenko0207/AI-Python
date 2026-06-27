import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


def create_neural_network(input_dim, hidden_units, lr):
    """
    Create a neural network model with a specified number of hidden units.

    Args:
        input_dim (int): Number of input features.
        hidden_units (int): Number of units in the hidden layer.
        lr (float): Learning rate for the SGD optimizer.

    Returns:
        model (Sequential): Compiled neural network model.
    """
    model = Sequential(
        [
            # Додаємо прихований шар із заданою кількістю нейронів
            Dense(
                units=hidden_units,
                activation="relu",
                input_dim=input_dim,
                name="hidden_layer",
            ),
            # Вихідний шар для бінарної класифікації (1 нейрон, Sigmoid)
            Dense(units=1, activation="sigmoid", name="output_layer"),
        ]
    )

    # Компілюємо модель, передаючи поточну швидкість навчання (learning rate) в SGD
    model.compile(
        optimizer=SGD(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_neural_network(model, X_train, y_train, num_epochs, batch_size):
    """
    Train the neural network model on the training data.

    Args:
        model (Sequential): Compiled neural network model.
        X_train (ndarray): Training input features.
        y_train (ndarray): Training labels.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        history: History object containing training metrics.
    """
    # verbose=0 вимикає довгі прогрес-бари, щоб не забивати термінал
    history = model.fit(
        X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0
    )
    return history


# 1. Генеруємо випадкові бінарні дані для демонстрації (1000 прикладів, 4 ознаки)
np.random.seed(42)
X_train = np.random.randn(1000, 4)
y_train = np.random.randint(0, 2, size=(1000, 1))

# 2. Визначаємо сітку гіперпараметрів для експериментів за порадою з методички
learning_rates = [0.001, 0.01, 0.1]
hidden_units_options = [4, 8, 16]

print("=" * 65)
print(" STARTING HYPERPARAMETER TUNING EXPERIMENTS")
print("=" * 65)

# 3. Цикл перебору всіх можливих комбінацій гіперпараметрів
experiment_num = 1
for lr in learning_rates:
    for hidden_units in hidden_units_options:
        print(f"\n[Experiment {experiment_num}] Testing Configuration:")
        print(f" -> Learning Rate: {lr} | Hidden Units: {hidden_units}")

        # Створення нової моделі під поточні параметри
        model = create_neural_network(input_dim=4, hidden_units=hidden_units, lr=lr)

        # Навчання моделі (20 епох, батч 32)
        history = train_neural_network(
            model, X_train, y_train, num_epochs=20, batch_size=32
        )

        # Витягуємо фінальні метрики після останньої епохи
        final_loss = history.history["loss"][-1]
        final_acc = history.history["accuracy"][-1]

        print(
            f"   Result setelah 20 Epochs -> Final Loss: {final_loss:.4f} | Final Accuracy: {final_acc * 100:.2f}%"
        )
        print("-" * 50)

        experiment_num += 1

print("\nAll experiments finished successfully!")
