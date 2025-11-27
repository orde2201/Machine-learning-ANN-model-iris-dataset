import numpy as np
from tensorflow.keras.models import load_model
import joblib

# 1. Load model
model = load_model("iris_ann_model.h5")

# 2. Load scaler
scaler = joblib.load("scaler.save")

# 3. Input data baru (contoh)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# 4. Normalisasi input (HARUS! Model ANN butuh input yang sama seperti saat training)
sample_scaled = scaler.transform(sample)

# 5. Prediksi
pred = model.predict(sample_scaled)

# 6. Menentukan kelas dari output one-hot
classes = ["Setosa", "Versicolor", "Virginica"]
predicted_class = classes[np.argmax(pred)]

print("Output Raw:", pred)
print("Prediksi:", predicted_class)
