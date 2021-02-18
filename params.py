import tensorflow as tf
import numpy as np

from tensorboard.plugins.hparams import api as hp

from dataset import get_dataset, prepare_dataset
from model import get_model
from helpers.decode_with_vocab import decode_with_vocab

device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
  raise SystemError("GPU device not found")

dataset = get_dataset("fr-en")

train_dataset = dataset[0:200000]

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
  max_window_size = 65
)

x_train = [np.array(encoder_input[0:100000]), np.array(decoder_input[0:100000])]
y_train = np.array(decoder_output[0:100000])

x_test = [np.array(encoder_input[100000:150000]), np.array(decoder_input[100000:150000])]
y_test = np.array(decoder_output[100000:150000])

BATCH_SIZE = hp.HParam("batch_num", hp.Discrete([32, 16]))
DENSE_NUM = hp.HParam("dense_num", hp.Discrete([512, 256]))
HEAD_NUM = hp.HParam("head_num", hp.Discrete([8, 4]))
EMBED_NUM = hp.HParam("embed_num", hp.Discrete([512, 256]))
LAYER_NUM = hp.HParam("layer_num", hp.Discrete([6]))

with tf.summary.create_file_writer("logs/hparam_tuning").as_default(): # ./tensorboard --logdir logs/
  hp.hparams_config(
    hparams=[LAYER_NUM, HEAD_NUM, EMBED_NUM, DENSE_NUM, BATCH_SIZE],
    metrics=[
      hp.Metric("accuracy"),
      hp.Metric("loss"),
      hp.Metric("val_accuracy"),
      hp.Metric("val_loss")
    ],
  )

def train_test_model(hparams):
  transformer_model = get_model(
    EMBEDDING_SIZE = hparams[EMBED_NUM],
    ENCODER_WINDOW_SIZE = 65,
    DECODER_WINDOW_SIZE = 65,
    ENCODER_VOCAB_SIZE = len(encoder_vocab),
    DECODER_VOCAB_SIZE = len(decoder_vocab),
    ENCODER_LAYERS = hparams[LAYER_NUM],
    DECODER_LAYERS = hparams[LAYER_NUM],
    NUMBER_HEADS = hparams[HEAD_NUM],
    DENSE_LAYER_SIZE = hparams[DENSE_NUM]
  )

  transformer_model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
  )

  transformer_model.fit(x_train, y_train, epochs = 2, batch_size = hparams[BATCH_SIZE])

  _, accuracy = transformer_model.evaluate(x_test, y_test)

  return accuracy


################################################
#                                              #
#                   HParams.                   #
#                                              #
################################################
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step = 1)

session_num = 0

for batch_num in BATCH_SIZE.domain.values:
  for dense_num in DENSE_NUM.domain.values:
    for num_heads in HEAD_NUM.domain.values:
      for num_embed in EMBED_NUM.domain.values:
        for num_units in LAYER_NUM.domain.values:
          hparams = {
              BATCH_SIZE: batch_num,
              DENSE_NUM: dense_num,
              HEAD_NUM: num_heads,
              EMBED_NUM: num_embed,
              LAYER_NUM: num_units
          }
          run_name = "run-%d" % session_num

          print("--- Starting trial: %s" % run_name)
          print({ h.name: hparams[h] for h in hparams })
          run("logs/hparam_tuning/" + run_name, hparams)

          session_num += 1
