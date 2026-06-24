import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(file_path):
    """
    Load the dataset from the specified file path.
    """
    # Завантажуємо файл з явним зазначенням роздільника-коми
    dataset = pd.read_csv(file_path, sep=",")
    return dataset


def inspect_data(dataset, num_rows=5):
    """
    Display the first few rows of the dataset.
    """
    print(f"\n--- Перші {num_rows} рядків датасету ---")
    print(dataset.head(num_rows))
    return None


def calculate_summary_statistics(dataset):
    """
    Calculate summary statistics for numerical columns in the dataset.
    """
    print("\n--- Описова статистика (Summary Statistics) ---")
    summary_stats = dataset.describe()
    print(summary_stats)
    return summary_stats


def visualize_data(dataset):
    """
    Create visualizations to explore the dataset and save them as images.
    """
    # Виокремлюємо тільки числові колонки для графіків
    numeric_cols = dataset.select_dtypes(include=["number"]).columns

    # 1. Гістограми для числових колонок
    print("\n[Візуалізація] Зберігаємо гістограми розподілу...")
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(dataset[col], kde=True, bins=30)
        plt.title(f"Розподіл показника: {col}")
        plt.xlabel(col)
        plt.ylabel("Частота")
        plt.tight_layout()

        # Автоматичне збереження у форматі PNG (наприклад, open_distribution.png)
        plt.savefig(f"{col}_distribution.png", dpi=300)
        plt.close()  # Закриваємо фігуру, щоб не накопичувати пам'ять

    # 2. Матриця кореляції (Correlation heatmap)
    print("[Візуалізація] Зберігаємо теплову карту кореляції...")
    plt.figure(figsize=(10, 8))
    correlation_matrix = dataset.corr(numeric_only=True)
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Матриця кореляції числових ознак")
    plt.tight_layout()

    # Збереження матриці
    plt.savefig("correlation_heatmap.png", dpi=300)
    plt.close()

    print("[Успіх] Усі графіки збережено в папку проекту.")
    return None


def handle_missing_values(dataset):
    """
    Handle missing values in the dataset.
    """
    print("\n--- Перевірка на пропущені значення ---")
    missing_count = dataset.isnull().sum()
    print(
        missing_count[missing_count > 0]
        if missing_count.sum() > 0
        else "Пропусків не знайдено."
    )

    # Створюємо копію, щоб не псувати оригінальний датафрейм
    dataset_filled = dataset.copy()

    # Визначаємо числові колонки
    numeric_cols = dataset_filled.select_dtypes(include=["number"]).columns

    # Заповнюємо пропуски середнім значенням по кожній колоночці
    for col in numeric_cols:
        mean_value = dataset_filled[col].mean()
        dataset_filled[col] = dataset_filled[col].fillna(mean_value)

    return dataset_filled


# --- ЗАПУСК АНАЛІЗУ ---
# Вказуємо ваш реальний файл замість стандартного шаблону
file_path = "brent_prices.csv"
data = load_dataset(file_path)

# Виклик функцій дослідження даних
inspect_data(data)
summary_stats = calculate_summary_statistics(data)
visualize_data(data)
data_filled = handle_missing_values(data)
