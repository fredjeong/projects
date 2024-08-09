
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.optimizers import Adam


def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse'):
    """ A multi-layer perceptron """
    print(n_action)
    
    model = tf.keras.Sequential()
    #
    model.add(tf.keras.Input((None, 3))) # Input shape 찾기
    #
    model.add(tf.keras.layers.Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
    for _ in range(n_hidden_layer):
        model.add(tf.keras.Dense(n_neuron_per_layer, activation=activation))
    model.add(tf.keras.Dense(n_action, activation='linear'))
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam())
    print(model.summary())
    return model