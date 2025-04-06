
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import os

def build_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_save_model(X_train, y_train, model_path='depression_model.h5', epochs=50, batch_size=16):
    model = build_ann_model(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(model_path)
    print(f"[✔] Đã lưu mô hình tại: {model_path}")
    return model


def load_ann_model(model_path='depression_model.h5'):
    if os.path.exists(model_path):
        print(f"[✓] Đang tải mô hình từ: {model_path}")
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_path}")


def predict_depression(model, X_input):
    probabilities = model.predict(X_input)
    return np.squeeze(probabilities)  
