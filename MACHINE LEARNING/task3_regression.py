import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_dataset(file_path):
    """
    Load a dataset from a CSV file.
    """
    try:
        # Намагаємось завантажити реальний файл
        dataset = pd.read_csv(file_path)
    except FileNotFoundError:
        # Якщо файлу немає, створюємо тестовий набір на місці
        print(f"[Warning] {file_path} not found. Generating sample data...")
        np.random.seed(42)
        n_samples = 500
        sq_ft = np.random.normal(1500, 500, n_samples).round()
        beds = np.random.choice([1, 2, 3, 4, 5], size=n_samples)
        price = (sq_ft * 150) + (beds * 25000) + np.random.normal(0, 15000, n_samples)

        dataset = pd.DataFrame(
            {"square_footage": sq_ft, "num_bedrooms": beds, "price": price}
        )
        # Зберігаємо його, щоб він з'явився в папці
        dataset.to_csv(file_path, index=False)

    return dataset


def preprocess_data(dataset):
    """
    Preprocess the dataset by selecting relevant features and target.
    """
    # Видаляємо пропуски, якщо вони є в оригінальному файлі
    dataset = dataset.dropna(subset=["square_footage", "num_bedrooms", "price"])

    # Виділяємо фічі та цільову змінну згідно з докстрінгом
    X = dataset[["square_footage", "num_bedrooms"]]
    y = dataset["price"]

    return X, y


def train_regression_model(X_train, y_train):
    """
    Train a regression model using the provided training data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(model, X_val, y_val):
    """
    Evaluate the performance of a regression model on validation data.
    """
    y_pred = model.predict(X_val)

    # Розрахунок метрик
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r_squared = r2_score(y_val, y_pred)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R-squared": r_squared}


# ==========================================
# EXECUTION FLOW
# ==========================================

# Load the dataset
file_path = "House_prices.csv"  # Назва файлу згідно з шаблоном
data = load_dataset(file_path)

# Preprocess the data
X, y = preprocess_data(data)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = train_regression_model(X_train, y_train)

# Evaluate the model
evaluation_metrics = evaluate_regression_model(model, X_val, y_val)
print("Evaluation Metrics:", evaluation_metrics)
