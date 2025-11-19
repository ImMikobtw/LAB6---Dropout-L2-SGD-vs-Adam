# LAB6
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

print("Train:", x_train.shape, y_train.shape)
print("Test:",  x_test.shape,  y_test.shape)

def build_model(dropout_rate=0.3, l2_rate=0.0, optimizer="adam"):
    return keras.Sequential([
        layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_rate)),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_rate)),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation="softmax")
    ]), optimizer

print("\n================ Dropout = 0.3 ================")
model_03, opt = build_model(dropout_rate=0.3)
model_03.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history_03 = model_03.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=2)

loss_03, acc_03 = model_03.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy (Dropout 0.3): {acc_03:.4f}")

print("\n================ Dropout = 0.7 ================")
model_07, opt = build_model(dropout_rate=0.7)
model_07.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history_07 = model_07.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=2)

loss_07, acc_07 = model_07.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy (Dropout 0.7): {acc_07:.4f}")

print("\n================ Adam optimizer ================")
model_adam, _ = build_model(dropout_rate=0.3)
model_adam.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_adam.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=2)
adam_loss, adam_acc = model_adam.evaluate(x_test, y_test, verbose=0)
print(f"Adam Accuracy: {adam_acc:.4f}")


print("\n================ SGD optimizer ================")
model_sgd, _ = build_model(dropout_rate=0.3)
model_sgd.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
model_sgd.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=2)
sgd_loss, sgd_acc = model_sgd.evaluate(x_test, y_test, verbose=0)
print(f"SGD Accuracy: {sgd_acc:.4f}")

print("\n=============== L2 = 0.01 ===============")
model_l2_001, _ = build_model(dropout_rate=0.3, l2_rate=0.01)
model_l2_001.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_l2_001.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=2)
loss001, acc001 = model_l2_001.evaluate(x_test, y_test, verbose=0)
print(f"L2=0.01 Accuracy: {acc001:.4f}")


print("\n=============== L2 = 0.0001 ===============")
model_l2_0001, _ = build_model(dropout_rate=0.3, l2_rate=0.0001)
model_l2_0001.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_l2_0001.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=2)
loss0001, acc0001 = model_l2_0001.evaluate(x_test, y_test, verbose=0)
print(f"L2=0.0001 Accuracy: {acc0001:.4f}")


print("\n==================== DONE ====================")
