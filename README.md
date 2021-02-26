Requirements

```
$ cat requirements.txt && pip install -r requirements.txt
```

```
tensorflow==2.3.0
nump==1.18.5
nltk==3.5
```

```
$ tensorboard --version
> 2.4.1

$ pip --version
> pip 21.0.1 from /Users/ays-dev/opt/anaconda3/envs/tf/lib/python3.7/site-packages/pip (python 3.7)

$ python --version
> Python 3.7.9
```


Train

```python
import tensorflow as tf
import numpy as np

from dataset import get_dataset, prepare_dataset
from model import get_model


dataset = get_dataset("fr-en")

print("Dataset loaded. Length:", len(dataset), "lines")

train_dataset = dataset[0:100000]

print("Train data loaded. Length:", len(train_dataset), "lines")

(encoder_input,
decoder_input,
decoder_output,
encoder_vocab,
decoder_vocab,
encoder_inverted_vocab,
decoder_inverted_vocab) = prepare_dataset(
  train_dataset,
  shuffle = False,
  lowercase = True,
  max_window_size = 20
)

transformer_model = get_model(
  EMBEDDING_SIZE = 64,
  ENCODER_VOCAB_SIZE = len(encoder_vocab),
  DECODER_VOCAB_SIZE = len(decoder_vocab),
  ENCODER_LAYERS = 2,
  DECODER_LAYERS = 2,
  NUMBER_HEADS = 4,
  DENSE_LAYER_SIZE = 128
)

transformer_model.compile(
  optimizer = "adam",
  loss = [
    "sparse_categorical_crossentropy"
  ],
  metrics = [
    "accuracy"
  ]
)

transformer_model.summary()

x = [np.array(encoder_input), np.array(decoder_input)]
y = np.array(decoder_output)

name = "transformer"
checkpoint_filepath = "./logs/transformer_ep-{epoch:02d}_loss-{loss:.2f}_acc-{accuracy:.2f}.ckpt"

tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir = "logs/{}".format(name)
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath = checkpoint_filepath,
  monitor = "loss",
  mode = "min",
  verbose = True,
  save_weights_only = True
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
  monitor = "loss",
  mode = "min",
  patience = 2,
  min_delta = 0.001
)

transformer_model.fit(
  x,
  y,
  epochs = 10,
  batch_size = 32,
  callbacks=[
    tensorboard_callback,
    model_checkpoint_callback,
    early_stopping_callback
  ]
)
```


Output

```
Dataset loaded. Length: 185583 lines
Train data loaded. Length: 100000 lines
Model: "Transformer-Model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
Encoder-Input (InputLayer)      [(None, None)]       0
__________________________________________________________________________________________________
Decoder-Input (InputLayer)      [(None, None)]       0
__________________________________________________________________________________________________
Encoder-Word-Embedding (WordEmb (None, None, 64)     1110016     Encoder-Input[0][0]
__________________________________________________________________________________________________
Decoder-Word-Embedding (WordEmb (None, None, 64)     579456      Decoder-Input[0][0]
__________________________________________________________________________________________________
Encoder-Position-Embedding (Pos (None, None, 64)     0           Encoder-Word-Embedding[0][0]
__________________________________________________________________________________________________
Decoder-Position-Embedding (Pos (None, None, 64)     0           Decoder-Word-Embedding[0][0]
__________________________________________________________________________________________________
Encoder (Encoder)               (None, None, 64)     66688       Encoder-Position-Embedding[0][0]
__________________________________________________________________________________________________
Decoder (Decoder)               (None, None, 64)     66688       Decoder-Position-Embedding[0][0]
__________________________________________________________________________________________________
Decoder-Output (Dense)          (None, None, 9054)   588510      Decoder[0][0]
==================================================================================================
Total params: 2,411,358
Trainable params: 2,411,358
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/10
3125/3125 [==============================] - ETA: 0s - loss: 1.0576 - accuracy: 0.8354
Epoch 00001: saving model to ./logs/transformer_ep-01_loss-1.06_acc-0.84.ckpt
3125/3125 [==============================] - 888s 284ms/step - loss: 1.0576 - accuracy: 0.8354
Epoch 2/10
3125/3125 [==============================] - ETA: 0s - loss: 0.4799 - accuracy: 0.9101
Epoch 00002: saving model to ./logs/transformer_ep-02_loss-0.48_acc-0.91.ckpt
3125/3125 [==============================] - 900s 288ms/step - loss: 0.4799 - accuracy: 0.9101
Epoch 3/10
3125/3125 [==============================] - ETA: 0s - loss: 0.3348 - accuracy: 0.9308
Epoch 00003: saving model to ./logs/transformer_ep-03_loss-0.33_acc-0.93.ckpt
3125/3125 [==============================] - 904s 289ms/step - loss: 0.3348 - accuracy: 0.9308
...
Epoch 9/10
3125/3125 [==============================] - ETA: 0s - loss: 0.1527 - accuracy: 0.9603
Epoch 00009: saving model to ./logs/transformer_ep-09_loss-0.15_acc-0.96.ckpt
3125/3125 [==============================] - 878s 281ms/step - loss: 0.1527 - accuracy: 0.9603
Epoch 10/10
3125/3125 [==============================] - ETA: 0s - loss: 0.1426 - accuracy: 0.9624
Epoch 00010: saving model to ./logs/transformer_ep-10_loss-0.14_acc-0.96.ckpt
3125/3125 [==============================] - 877s 281ms/step - loss: 0.1426 - accuracy: 0.9624
```


Predict

```python
import tensorflow as tf
import numpy as np

from dataset import get_dataset, prepare_dataset
from model import get_model
from utils.make_translate import make_translate


dataset = get_dataset("fr-en")

print("Dataset loaded. Length:", len(dataset), "lines")

train_dataset = dataset[0:100000]

print("Train data loaded. Length:", len(train_dataset), "lines")

(encoder_input,
decoder_input,
decoder_output,
encoder_vocab,
decoder_vocab,
encoder_inverted_vocab,
decoder_inverted_vocab) = prepare_dataset(
  train_dataset,
  shuffle = False,
  lowercase = True,
  max_window_size = 20
)

transformer_model = get_model(
  EMBEDDING_SIZE = 64,
  ENCODER_VOCAB_SIZE = len(encoder_vocab),
  DECODER_VOCAB_SIZE = len(decoder_vocab),
  ENCODER_LAYERS = 2,
  DECODER_LAYERS = 2,
  NUMBER_HEADS = 4,
  DENSE_LAYER_SIZE = 128
)

transformer_model.summary()

transformer_model.load_weights('./logs/transformer_ep-10_loss-0.14_acc-0.96.ckpt')

translate = make_translate(transformer_model, encoder_vocab, decoder_vocab, decoder_inverted_vocab)

translate("c'est une belle journée .")
translate("j'aime manger du gâteau .")
translate("c'est une bonne chose .")
translate("il faut faire à manger pour nourrir les gens .")
translate("tom a acheté un nouveau vélo .")
```

```
Original: c'est une belle journée .
Traduction: it' s a beautiful day .
Original: j'aime manger du gâteau .
Traduction: i like to eat some cake .
Original: c'est une bonne chose .
Traduction: that' s a good thing .
Original: il faut faire à manger pour nourrir les gens .
Traduction: we have to feed the people .
Original: tom a acheté un nouveau vélo .
Traduction: tom bought a new bicycle .
```


Fine-tuning

```python
import tensorflow as tf
import numpy as np

from tensorboard.plugins.hparams import api as hp

from dataset import get_dataset, prepare_dataset
from model import get_model


dataset = get_dataset("fr-en")

train_dataset = dataset[0:150]

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
  max_window_size = 20
)

x_train = [np.array(encoder_input[0:100]), np.array(decoder_input[0:100])]
y_train = np.array(decoder_output[0:100])

x_test = [np.array(encoder_input[100:150]), np.array(decoder_input[100:150])]
y_test = np.array(decoder_output[100:150])

BATCH_SIZE = hp.HParam("batch_num", hp.Discrete([32, 16]))
DENSE_NUM = hp.HParam("dense_num", hp.Discrete([512, 256]))
HEAD_NUM = hp.HParam("head_num", hp.Discrete([8, 4]))
EMBED_NUM = hp.HParam("embed_num", hp.Discrete([512, 256]))
LAYER_NUM = hp.HParam("layer_num", hp.Discrete([6, 4]))

with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
  hp.hparams_config(
    hparams=[LAYER_NUM, HEAD_NUM, EMBED_NUM, DENSE_NUM, BATCH_SIZE],
    metrics=[
      hp.Metric("val_accuracy")
    ],
  )

def train_test_model(hparams):
  transformer_model = get_model(
    EMBEDDING_SIZE = hparams[EMBED_NUM],
    ENCODER_VOCAB_SIZE = len(encoder_vocab),
    DECODER_VOCAB_SIZE = len(decoder_vocab),
    ENCODER_LAYERS = hparams[LAYER_NUM],
    DECODER_LAYERS = hparams[LAYER_NUM],
    NUMBER_HEADS = hparams[HEAD_NUM],
    DENSE_LAYER_SIZE = hparams[DENSE_NUM]
  )

  transformer_model.compile(
    optimizer = "adam",
    loss = ["sparse_categorical_crossentropy"],
    metrics = ["accuracy"]
  )

  transformer_model.fit(x_train, y_train, epochs = 1, batch_size = hparams[BATCH_SIZE])

  _, accuracy = transformer_model.evaluate(x_test, y_test)

  return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)
    accuracy = train_test_model(hparams)
    tf.summary.scalar("val_accuracy", accuracy, step = 1)

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
```

```
--- Starting trial: run-0
{'batch_num': 16, 'dense_num': 256, 'head_num': 4, 'embed_num': 256, 'layer_num': 4}
7/7 [==============================] - 1s 136ms/step - loss: 2.3796 - accuracy: 0.5590
2/2 [==============================] - 0s 35ms/step - loss: 1.3560 - accuracy: 0.8240
--- Starting trial: run-1
{'batch_num': 16, 'dense_num': 256, 'head_num': 4, 'embed_num': 256, 'layer_num': 6}
7/7 [==============================] - 1s 203ms/step - loss: 2.9393 - accuracy: 0.5625
2/2 [==============================] - 0s 53ms/step - loss: 1.1636 - accuracy: 0.8240
...
--- Starting trial: run-29
{'batch_num': 32, 'dense_num': 512, 'head_num': 8, 'embed_num': 256, 'layer_num': 6}
4/4 [==============================] - 1s 323ms/step - loss: 6.8676 - accuracy: 0.2650
2/2 [==============================] - 0s 74ms/step - loss: 13.1447 - accuracy: 0.0500
--- Starting trial: run-30
{'batch_num': 32, 'dense_num': 512, 'head_num': 8, 'embed_num': 512, 'layer_num': 4}
4/4 [==============================] - 2s 513ms/step - loss: 7.4851 - accuracy: 0.2650
2/2 [==============================] - 0s 111ms/step - loss: 13.9821 - accuracy: 0.0310
--- Starting trial: run-31
{'batch_num': 32, 'dense_num': 512, 'head_num': 8, 'embed_num': 512, 'layer_num': 6}
4/4 [==============================] - 3s 761ms/step - loss: 7.1519 - accuracy: 0.2660
2/2 [==============================] - 0s 162ms/step - loss: 14.0015 - accuracy: 0.0500
```

Visualize

```
$ ./tensorboard --logdir=./logs
```


Configs

```
Base:
  EMBEDDING_SIZE = 64,
  ENCODER_VOCAB_SIZE = 10000,
  DECODER_VOCAB_SIZE = 10000,
  ENCODER_LAYERS = 2,
  DECODER_LAYERS = 2,
  NUMBER_HEADS = 4,
  DENSE_LAYER_SIZE = 128
  MAX_WINDOW_SIZE = 20
  DATASET_SIZE = 100000
  BATCH_SIZE = 32
 -> 2.4 millions params (model size : ~28 Mo)

Big:
  EMBEDDING_SIZE = 512,
  ENCODER_VOCAB_SIZE = 30000,
  DECODER_VOCAB_SIZE = 30000,
  ENCODER_LAYERS = 6,
  DECODER_LAYERS = 6,
  NUMBER_HEADS = 8,
  DENSE_LAYER_SIZE = 1024
  MAX_WINDOW_SIZE = 65
  DATASET_SIZE = 200000
  BATCH_SIZE = 32
 -> 60 millions params (model size : ~600 Mo)
```

Credits

<pre>
<b>Coder un Transformer avec Tensorflow et Keras (LIVE)</b>
<a href="https://www.youtube.com/watch?v=mWA-PmxMBDk">https://www.youtube.com/watch?v=mWA-PmxMBDk</a>
<i>Thibault Neveu</i>
<a href="https://colab.research.google.com/drive/1akAsUAddF2-x57BJBA_gF-v4htyiR12y?usp=sharing">https://colab.research.google.com/drive/1akAsUAddF2-x57BJBA_gF-v4htyiR12y?usp=sharing</a>
</pre>

<pre>
<b>[TUTORIAL + CÓDIGO] Machine Translation usando redes TRANSFORMER (Python + Keras)</b>
<a href="https://www.youtube.com/watch?v=p2sTJYoIwj0">https://www.youtube.com/watch?v=p2sTJYoIwj0</a>
<i>codificandobits</i>
<a href="https://github.com/codificandobits/Traductor_con_redes_Transformer/blob/master/machine-translation-transformers.ipynb">https://github.com/codificandobits/Traductor_con_redes_Transformer/blob/master/machine-translation-transformers.ipynb</a>
</pre>

<pre>
<a href="https://github.com/CyberZHG/keras-transformer">https://github.com/CyberZHG/keras-transformer</a>
<i>CyberZHG</i>
</pre>
