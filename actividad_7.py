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
    loss="mean_squared_error"
                    )

#introducir datos
datos_carbon = [[1],[2],[3],[4],[5],[6],[7]]
datos_dureza = [[2],[4],[6],[8],[10],[12],[14]]
datos_carbon = np.array(datos_carbon)
datos_dureza = np.array(datos_dureza)

# datos_acero = pd.read_csv("/content/carbonpercentage_hardness.csv")
# print(datos_acero.head())
# datos_dureza = datos_acero.iloc[:,1:2].values
# datos_carbon = datos_acero.iloc[:,0:1].values
# datos_carbon = np.array(datos_carbon)
# datos_dureza = np.array(datos_dureza)
#print(datos_dureza)
#print(datos_carbon)


##entrenar modelo
entrenamiento_modelo = red_neuronal.fit(
    datos_carbon, #entradas
    datos_dureza, #salidas
    epochs=250,
    verbose=1)

# ver peso y sesgo aprendidos
weights, bias = red_neuronal.layers[0].get_weights()
print(f"peso aprendido: {weights[0][0]}")
print(f"sesgo aprendido: {bias[0]}")


#predecir valores
X_test = np.array([[.1],[.15],[.10],[.15],[1.00]], dtype=float)
predictions = red_neuronal.predict(X_test)

print("predicciones:")
for i, val in enumerate(X_test):
    print(f"x = {val[0]} -> y_pred = {predictions[i][0]}")
