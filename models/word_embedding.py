import tensorflow as tf

def mask_zero(x):
    mask = tf.greater(x, 0)
    mask = tf.cast(mask, dtype = tf.float32)
    return mask

class WordEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embedding_size, **kwargs):
    super().__init__(**kwargs)

    self.vocab_size = vocab_size
    self.embedding_size = embedding_size

  def get_config(self):
    config = super().get_config().copy()

    config.update({
      'vocab_size': self.vocab_size,
      'embedding_size': self.embedding_size
    })

    return config

  def build(self, input_shape):
    super().build(input_shape)

    self.word_embedding = tf.keras.layers.Embedding(
      self.vocab_size,
      self.embedding_size,
      mask_zero = True
    )

  def call(self, x):
    word_embedding = self.word_embedding(x)

    return word_embedding
