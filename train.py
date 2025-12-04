import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ------------------------------------------------------------------
# 1) Load Iris CSV dari Kaggle
# ------------------------------------------------------------------
df = pd.read_csv("IRIS.csv")

# ubah setiap spesies yang ada di header file IRIS.csv menjadi angka karna lebih mudah diolah daripada text
label_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
# untuk mengubah seluruh spesies menjadi angka
df['species'] = df['species'].map(label_map)

# Pisahkan fitur dan label, x untuk fitur, y untuk label
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['species'].values
target_names = ['setosa', 'versicolor', 'virginica']

# ------------------------------------------------------------------
# 3) Preprocessing (Split dan Scaling)
# ------------------------------------------------------------------

# Split train-test (STRATIFY HARUS PAKAI y, bukan y_cat!)
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode setelah split
y_train = to_categorical(y_train_raw)
y_test = to_categorical(y_test_raw)

# Scaling fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4) Bangun model ANN
# ------------------------------------------------------------------
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(12, activation='relu'),
    Dropout(0.1),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------------------
# 5) Train model
# ------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=8,
    verbose=1
)

# ------------------------------------------------------------------
# 6) Evaluasi
# ------------------------------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nEvaluasi test set -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Prediksi
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# ------------------------------------------------------------------
# 7) Visualisasi training
# ------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ------------------------------------------------------------------
# 8) Simpan model dan scaler
# ------------------------------------------------------------------
model.save("iris_ann_model.h5")

import joblib
joblib.dump(scaler, "scaler.save")

print("Model dan scaler disimpan: iris_ann_model.h5, scaler.save")

# ------------------------------------------------------------------
# 9) Contoh prediksi manual
# ------------------------------------------------------------------
def predict_sample(sample_array):
    """sample_array: list atau np.array panjang 4 -> fitur mentah"""
    arr = np.array(sample_array).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prob = model.predict(arr_scaled)[0]
    cls = np.argmax(prob)
    return target_names[cls], prob

example = [5.1, 3.5, 1.4, 0.2]
print("Contoh prediksi:", predict_sample(example))
