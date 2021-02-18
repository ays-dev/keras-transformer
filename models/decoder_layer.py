import tensorflow as tf

from .multi_head_attention import MultiHeadAttention

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_size, dense_layer_size, nb_head, **kwargs):
    super(**kwargs).__init__()

    self.embedding_size = embedding_size
    self.dense_layer_size = dense_layer_size
    self.nb_head = nb_head

  def build(self, input_shape):
    super().build(input_shape)

    self.attention = MultiHeadAttention(self.embedding_size, self.nb_head)
    self.norm = tf.keras.layers.LayerNormalization()
    self.dense_1 = tf.keras.layers.Dense(self.dense_layer_size)
    self.dense_2 = tf.keras.layers.Dense(self.embedding_size)

  def call(self, x):
    encoder_output, output_embedding = x

    self_attention = self.attention((output_embedding, output_embedding, output_embedding), mask = True)
    post_self_attention = self.norm(self_attention + output_embedding)

    encoder_attention = self.attention((post_self_attention, encoder_output, encoder_output))
    post_encoder_attention = self.norm(encoder_attention + post_self_attention)

    dense_out = self.dense_1(post_encoder_attention)
    dense_out = self.dense_2(dense_out)

    encoder_output = self.norm(dense_out + post_encoder_attention)

    return encoder_output
