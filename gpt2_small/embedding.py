import tensorflow as tf

class WTE(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__(name="wte")
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            name="embedding"
        )

    def call(self, x):
        return self.embedding(x)


class WPE(tf.keras.layers.Layer):
    def __init__(self, max_position: int, d_model: int) -> None:
        super().__init__(name="wpe")
        self.embedding = tf.keras.layers.Embedding(
            input_dim=max_position,
            output_dim=d_model,
            name="embeddings"
        )

    def call(self, positions):
        return self.embedding(positions)