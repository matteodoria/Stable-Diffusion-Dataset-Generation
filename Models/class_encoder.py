from tensorflow import keras

class ClassEncoder(keras.Model):
    def __init__(self, max_length, weight_path):
        input_class = keras.layers.Input(shape=(max_length,), dtype="float32", name="tokens")

        x = keras.layers.Dense(units=500)(input_class)

        x = keras.layers.Dense(units=76800, use_bias=False)(x)

        embedded = keras.layers.Reshape(target_shape=(100, 768))(x)

        super().__init__(input_class, embedded, name="MyEmbedding")

        try:
            print("... Loading Embedding model weights from {} ...".format(weight_path))
            self.load_weights(weight_path)
            print("DONE")
        except FileNotFoundError:
            print("No weights found at specified path. Training Embedding Model from scratch.")
        except Exception as e:
            print(f"An error occurred while loading weights: {e}")
