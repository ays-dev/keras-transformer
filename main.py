import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.metrics import categorical_accuracy

import tensorflow as tf
import numpy as np
import os.path
import time

from dataset import get_dataset, prepare_dataset
from model import get_model
from helpers.decode_with_vocab import decode_with_vocab

# device_name = tf.test.gpu_device_name()
# if device_name != "/device:GPU:0":
#   raise SystemError("GPU device not found")

dataset = get_dataset("fr-en")

print("Dataset loaded. Length:", len(dataset), "lines")

train_dataset = dataset[0:15000]

print("Train data loaded. Length:", len(train_dataset), "lines")

(encoder_input,
decoder_input,
decoder_output,
encoder_vocab,
decoder_vocab,
encoder_inverted_vocab,
decoder_inverted_vocab) = prepare_dataset(
  train_dataset,
  shuffle = True,
  lowercase = True,
  max_window_size = 15
)

# print("encoder_input", encoder_input)
# print("decoder_input", decoder_input)
# print("decoder_output", decoder_output)
# print("encoder_vocab", encoder_vocab)
# print("decoder_vocab", decoder_vocab)
# print("decoder_inverted_vocab", decoder_inverted_vocab)

################################################
#                                              #
#              Custom Transformer              #
#                                              #
################################################
transformer_model = get_model(
  EMBEDDING_SIZE = 64,
  ENCODER_WINDOW_SIZE = 15,
  DECODER_WINDOW_SIZE = 15,
  ENCODER_VOCAB_SIZE = len(encoder_vocab),
  DECODER_VOCAB_SIZE = len(decoder_vocab),
  ENCODER_LAYERS = 1,
  DECODER_LAYERS = 1,
  NUMBER_HEADS = 2,
  DENSE_LAYER_SIZE = 64
)

transformer_model.compile(
  optimizer = "adam",
  loss = ["sparse_categorical_crossentropy"],
  metrics = ["accuracy"]
)

transformer_model.summary()

x = [np.array(encoder_input), np.array(decoder_input)]
y = np.array(decoder_output)

checkpoint_filepath = "./logs/checkpoint"
name = "transformer_model_categorical_cross_entropy" # ./tensorboard --logdir logs/

# if os.path.isfile(checkpoint_filepath + ".index"):
#   transformer_model.load_weights(checkpoint_filepath)
# else:
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "logs/{}".format(name))
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath = checkpoint_filepath,
  save_weights_only = True,
  monitor = "val_loss",
  mode = "min",
  save_best_only = True
)
custom_early_stopping = tf.keras.callbacks.EarlyStopping(
  monitor = "val_loss",
  patience = 2,
  min_delta = 0.001,
  mode="min"
)

transformer_model.fit(
  x,
  y,
  epochs = 10,
  batch_size = 16,
  validation_split = 0.25,
  callbacks=[
    model_checkpoint_callback,
    tensorboard_callback
  ]
)
# transformer_model.evaluate(x, y)

# transformer_model(x)
