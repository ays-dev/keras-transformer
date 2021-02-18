import tensorflow as tf

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embedding_size, **kwargs):
    super(**kwargs).__init__()

    self.vocab_size = vocab_size
    self.embedding_size = embedding_size

  def build(self, input_shape):
    super().build(input_shape)

    self.word_embedding = tf.keras.layers.Embedding(
      self.vocab_size,
      self.embedding_size
    )

  def call(self, x):
    word_embedding = self.word_embedding(x)

    return word_embedding
