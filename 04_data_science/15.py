from tensorflow import keras
from tensorflow.keras import layers

VOCAB_SIZE = 10000  # Keep the 10,000 most frequent words
MAX_LENGTH = 100    # Use the last 100 words of each review

keras.utils.set_random_seed(42)  # Same result on every run

# 1. Load 50,000 real movie reviews (IMDB, downloaded on first run). The reviews are already
#    tokenized: every word is replaced by an integer ID. Labels: 1 = positive, 0 = negative.
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

# 2. Pad or cut every review to the same length
x_train = keras.utils.pad_sequences(x_train, maxlen=MAX_LENGTH)
x_test = keras.utils.pad_sequences(x_test, maxlen=MAX_LENGTH)

# 3. Build the LSTM model
model = keras.Sequential([
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32),  # Word ID -> vector of 32 numbers
    layers.LSTM(32),                                        # Reads the review word by word
    layers.Dropout(0.5),                                    # Reduces overfitting
    layers.Dense(1, activation='sigmoid')                   # Probability that the review is positive
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train; stop early when the validation loss stops improving
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, callbacks=[early_stopping], verbose=2)

# 5. Evaluate on 25,000 reviews the model has never seen
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
baseline = max(y_test.mean(), 1 - y_test.mean())
print(f"Test accuracy: {accuracy:.2f} (baseline, always guessing one class: {baseline:.2f})")

# 6. Try it on new reviews
word_index = keras.datasets.imdb.get_word_index()


def encode_review(text):
    """Turns raw text into the same integer IDs the model was trained on."""
    words = text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').split()
    # IDs are shifted by 3 in this dataset: 0 = padding, 1 = start of review, 2 = unknown word
    ids = [1] + [word_index[w] + 3 if w in word_index and word_index[w] + 3 < VOCAB_SIZE else 2 for w in words]
    return keras.utils.pad_sequences([ids], maxlen=MAX_LENGTH)


for review in ["This movie was fantastic, I loved every minute of it.",
               "Terrible film. Boring plot and awful acting, a complete waste of time."]:
    score = model.predict(encode_review(review), verbose=0)[0][0]
    print(f"{score:.2f} positive -> {review}")
