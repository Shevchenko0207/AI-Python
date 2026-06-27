import numpy as np
import os

# Налаштовуємо Keras використовувати PyTorch бекенд
os.environ["KERAS_BACKEND"] = "torch"

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.models import load_model


# Проста та надійна реалізація токенізатора на чистому Python
class SimpleTokenizer:
    def __init__(self, max_words):
        self.max_words = max_words
        self.word_index = {}

    def fit_on_texts(self, texts):
        word_counts = {}
        for text in texts:
            # Приводимо до нижнього регістру та розбиваємо на слова
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Сортуємо за частотою використання
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Заповнюємо словник індексами (починаючи з 1, бо 0 зарезервовано для padding)
        for i, (word, _) in enumerate(sorted_words[: self.max_words - 1]):
            self.word_index[word] = i + 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            for word in text.lower().split():
                if word in self.word_index:
                    seq.append(self.word_index[word])
            sequences.append(seq)
        return sequences


def manual_pad_sequences(sequences, maxlen):
    """Нативний педдінг без залежності від keras_preprocessing"""
    padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        # Якщо послідовність довша за maxlen, обрізаємо її
        if len(seq) > maxlen:
            truncated = seq[-maxlen:]
        else:
            truncated = seq
        # Заповнюємо кінець масиву (post-padding або звичайне вирівнювання праворуч)
        padded[i, -len(truncated) :] = truncated
    return padded


def train_sentiment_analysis_model(
    texts, labels, max_words, embedding_dim, num_epochs, batch_size
):
    # 1. Використовуємо наш надійний токенізатор
    tokenizer = SimpleTokenizer(max_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Визначаємо максимальну довжину
    max_len = max(len(seq) for seq in sequences) if sequences else 10
    X_padded = manual_pad_sequences(sequences, maxlen=max_len)
    y = np.array(labels)

    # 2. Архітектура мережі
    model = Sequential(
        [
            Embedding(
                input_dim=max_words, output_dim=embedding_dim, name="embedding_layer"
            ),
            Flatten(name="flatten"),
            Dense(units=8, activation="relu", name="hidden_layer"),
            Dense(units=1, activation="sigmoid", name="output_layer"),
        ]
    )

    # 3. Компіляція
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("Старт навчання моделі Sentiment Analysis...")
    model.fit(X_padded, y, epochs=num_epochs, batch_size=batch_size, verbose=1)

    if not os.path.exists("models"):
        os.mkdir("models")
    model.save("models/sentiment_model.keras")
    print(
        "-> Модель успішно затреновано та збережено в 'models/sentiment_model.keras'\n"
    )

    return model, tokenizer, max_len


# ==========================================================
# Дані та запуск
# ==========================================================

X_train_texts = [
    "I absolutely loved this movie it was fantastic and wonderful",
    "This was the worst film I have ever seen totally boring",
    "Great acting amazing plot and brilliant direction highly recommend",
    "What a waste of time horrible acting and terrible script",
    "An outstanding masterpiece with beautiful cinematography",
    "I hated this show it was dynamic garbage and extremely bad",
]
y_train_labels = [1, 0, 1, 0, 1, 0]

MAX_WORDS = 1000
EMBEDDING_DIM = 16
NUM_EPOCHS = 15  # Трохи збільшимо для кращої збіжності
BATCH_SIZE = 2

# Тренування
model, tokenizer, max_len = train_sentiment_analysis_model(
    texts=X_train_texts,
    labels=y_train_labels,
    max_words=MAX_WORDS,
    embedding_dim=EMBEDDING_DIM,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
)

# Тестування збереженої моделі
print("--- Тестування збереженої моделі на нових відгуках ---")
trained_model = load_model("models/sentiment_model.keras")

new_reviews = [
    "This movie was wonderful and amazing highly recommend",
    "It was a terrible film and a complete waste of time",
]

# Токенізація та педдінг нових відгуків
new_sequences = tokenizer.texts_to_sequences(new_reviews)
new_padded = manual_pad_sequences(new_sequences, maxlen=max_len)

predictions = trained_model.predict(new_padded, verbose=0)

print("\nРезультати аналізу тональності:")
print("=" * 65)
for i, review in enumerate(new_reviews):
    binary_label = 1 if predictions[i][0] >= 0.5 else 0
    sentiment_str = "🟢 POSITIVE" if binary_label == 1 else "🔴 NEGATIVE"

    print(f"Review: '{review}'")
    print(f"Raw Score: {predictions[i][0]:.4f} -> Sentiment: {sentiment_str}")
    print("-" * 65)
