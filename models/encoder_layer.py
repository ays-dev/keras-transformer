import tensorflow as tf

from .multi_head_attention import MultiHeadAttention

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_size, dense_layer_size, nb_head, **kwargs):
    super().__init__(**kwargs)

    self.embedding_size = embedding_size
    self.dense_layer_size = dense_layer_size
    self.nb_head = nb_head

  def get_config(self):
    config = super().get_config().copy()

    config.update({
      'embedding_size': self.embedding_size,
      'dense_layer_size': self.dense_layer_size,
      'nb_head': self.nb_head
    })

    return config

  def build(self, input_shape):
    super().build(input_shape)

    self.attention = MultiHeadAttention(self.embedding_size, self.nb_head)
    self.norm_1 = tf.keras.layers.LayerNormalization()
    self.norm_2 = tf.keras.layers.LayerNormalization()
    self.dense_1 = tf.keras.layers.Dense(self.dense_layer_size)
    self.dense_2 = tf.keras.layers.Dense(self.embedding_size)

  def call(self, x):
    self_attention = self.attention((x, x, x))
    self_attention = self.norm_1(self_attention + x)

    dense_out = self.dense_1(self_attention)
    dense_out = self.dense_2(dense_out)

    encoder_output = self.norm_2(dense_out + self_attention)

    return encoder_output
