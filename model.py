import tensorflow as tf

from models.embedding import Embedding
from models.positional_embedding import PositionalEmbedding
from models.decoder import Decoder
from models.encoder import Encoder

def get_model(
    EMBEDDING_SIZE = 64,
    DENSE_LAYER_SIZE = 64,
    ENCODER_WINDOW_SIZE = 8,
    DECODER_WINDOW_SIZE = 8,
    ENCODER_VOCAB_SIZE = 12,
    DECODER_VOCAB_SIZE = 12,
    ENCODER_LAYERS = 1,
    DECODER_LAYERS = 1,
    NUMBER_HEADS = 1
  ):
  encoder_layer_input = tf.keras.Input(shape = ENCODER_WINDOW_SIZE)
  decoder_layer_input = tf.keras.Input(shape = DECODER_WINDOW_SIZE)

  encoder_embedding = Embedding(ENCODER_VOCAB_SIZE, EMBEDDING_SIZE)(encoder_layer_input)
  decoder_embedding = Embedding(DECODER_VOCAB_SIZE, EMBEDDING_SIZE)(decoder_layer_input)

  encoder_pos_encoding = PositionalEmbedding(ENCODER_WINDOW_SIZE, EMBEDDING_SIZE)(tf.range(ENCODER_WINDOW_SIZE))
  decoder_pos_encoding = PositionalEmbedding(DECODER_WINDOW_SIZE, EMBEDDING_SIZE)(tf.range(DECODER_WINDOW_SIZE))

  encoder_embedding = encoder_embedding + encoder_pos_encoding
  decoder_embedding = decoder_embedding + decoder_pos_encoding

  encoder_output = Encoder(ENCODER_LAYERS, EMBEDDING_SIZE, DENSE_LAYER_SIZE, NUMBER_HEADS)(encoder_embedding)
  decoder_output = Decoder(DECODER_LAYERS, EMBEDDING_SIZE, DENSE_LAYER_SIZE, NUMBER_HEADS)((decoder_embedding, encoder_output))

  output_predictions = tf.keras.layers.Dense(DECODER_VOCAB_SIZE)(decoder_output)
  predictions = tf.nn.softmax(output_predictions, axis = -1)

  model = tf.keras.Model([encoder_layer_input, decoder_layer_input], predictions)

  return model
