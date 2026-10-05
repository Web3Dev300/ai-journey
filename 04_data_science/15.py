import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and prepare dataset
texts = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "Absolutely fantastic service, highly recommend.",
    "I hate this item, it's terrible.",
    "Great quality and fast shipping.",
    "Not worth the money, very disappointed.",
    "Excellent customer support, very helpful.",
    "The product broke after one use, very poor quality.",
    "I'm extremely satisfied with my purchase.",
    "Terrible, I will never buy from this company again."
]
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) # 1 for positive sentiment, 0 for negative sentiment

# Preprocess text data
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=20, padding='post')

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=64))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data,labels, epochs=10, batch_size=2, validation_split=0.2)

# Evaluate the model
loss,accuracy = model.evaluate(data,labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Prediction (optional)
predictions = model.predict(data)
print(predictions)

