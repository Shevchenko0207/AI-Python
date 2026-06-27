import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    Sigmoid activation function.

    Args:
        x (ndarray): Input values.

    Returns:
        ndarray: Output of the sigmoid function.
    """
    # Захищаємося від можливого переповнення (overflow) за допомогою np.clip, як у твоїх лекціях
    x_clipped = np.clip(x, -250, 250)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.

    Args:
        x (ndarray): Input values.

    Returns:
        ndarray: Output of the ReLU function.
    """
    # Повертає x, якщо x > 0, інакше 0
    return np.maximum(0, x)


def tanh(x):
    """
    Hyperbolic tangent (tanh) activation function.

    Args:
        x (ndarray): Input values.

    Returns:
        ndarray: Output of the tanh function.
    """
    # Використовуємо вбудовану оптимізовану функцію NumPy
    return np.tanh(x)


# 1. Створюємо масив значень x від -5 до 5 (100 точок для плавності ліній)
x = np.linspace(-5, 5, 100)

# 2. Рахуємо значення y для кожної функції
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

# 3. Налаштовуємо графік за допомогою Matplotlib
plt.figure(figsize=(10, 6))

# Будуємо лінії з різними кольорами та стилями
plt.plot(x, y_sigmoid, label="Sigmoid", color="blue", linewidth=2)
plt.plot(x, y_tanh, label="Tanh", color="orange", linewidth=2, linestyle="--")
plt.plot(x, y_relu, label="ReLU", color="green", linewidth=2, linestyle="-.")

# Додаємо сітку та центральні осі для красивого відображення меж насичення
plt.axhline(0, color="black", linewidth=0.8, linestyle=":")
plt.axvline(0, color="black", linewidth=0.8, linestyle=":")

# 4. Додаємо підписи осі, заголовок та легенду
plt.xlabel("Input Value (x)", fontsize=12)
plt.ylabel("Output Value (y)", fontsize=12)
plt.title(
    "Comparison of Activation Functions: Sigmoid, Tanh, and ReLU",
    fontsize=14,
    fontweight="bold",
)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, alpha=0.3)

# Обмежуємо y, щоб графік ReLU не "забивав" масштаб інших функцій
plt.ylim(-1.5, 2.5)

# 5. Показуємо та зберігаємо графік
plt.tight_layout()
plt.savefig("activation_functions.png", dpi=300)
print("-> Графік успішно побудовано та збережено як 'activation_functions.png'")
plt.show()
