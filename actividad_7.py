import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib

print("version de Tensorflow",tf.__version__)
print("version de Keras", keras.__version__)
print("version de Numpy", np.__version__)
print("version de Pandas", pd.__version__)
print("version de Matplotlib", matplotlib.__version__)

#definir red neuronal
  # units=1               numero de neuronas
  # entradas              input_shape=[1]
  # activation="linear"   activacion (linear, relu, sigmoid, softmax, etc)
red_neuronal = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], activation="linear")
                                  ])
#print(red_neuronal.summary())


red_neuronal.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='mean_squared_error'
                    )

#introducir datos
datos_x = [[1],[2],[3],[4],[5],[6],[7]]
datos_y = [[2],[4],[6],[8],[10],[12],[14]]
datos_x = np.array(datos_x)
datos_y = np.array(datos_y)

##entrenar modelo
entrenamiento_modelo = red_neuronal.fit(
    datos_x,
    datos_y,
    epochs=250,
    verbose=1)

# ver peso y sesgo aprendidos
weights, bias = red_neuronal.layers[0].get_weights()
print(f"peso aprendido: {weights[0][0]}")
print(f"sesgo aprendido: {bias[0]}")


#predecir valores
X_test = np.array([[1],[5],[10],[15],[100]], dtype=float)
predictions = red_neuronal.predict(X_test)

print("Predicciones:")
for i, val in enumerate(X_test):
    print(f"x = {val[0]} -> y_pred = {predictions[i][0]}")