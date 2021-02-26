Requirements

```
$ cat requirements.txt
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

```
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
  dump_dic = True,
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

transformer_model.fit(
  x,
  y,
  epochs = 10,
  batch_size = 32,
  callbacks=[
    model_checkpoint_callback,
    tensorboard_callback
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
Encoder-Token-Embedding (Embedd (None, None, 64)     1110016     Encoder-Input[0][0]
__________________________________________________________________________________________________
Decoder-Token-Embedding (Embedd (None, None, 64)     579456      Decoder-Input[0][0]
__________________________________________________________________________________________________
Encoder-Pos-Embedding (PosEmbed (None, None, 64)     0           Encoder-Token-Embedding[0][0]
__________________________________________________________________________________________________
Decoder-Pos-Embedding (PosEmbed (None, None, 64)     0           Decoder-Token-Embedding[0][0]
__________________________________________________________________________________________________
Encoder (Encoder)               (None, None, 64)     66688       Encoder-Pos-Embedding[0][0]
__________________________________________________________________________________________________
Decoder (Decoder)               (None, None, 64)     66688       Decoder-Pos-Embedding[0][0]
                                                                 Encoder[0][0]
__________________________________________________________________________________________________
Decoder-Output (Dense)          (None, None, 9054)   588510      Decoder[0][0]
==================================================================================================
Total params: 2,411,358
Trainable params: 2,411,358
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/10
3125/3125 [==============================] - ETA: 0s - loss: 1.0451 - accuracy: 0.8375
Epoch 00001: saving model to ./logs/transformer_ep-01_loss-1.05_acc-0.84.ckpt
3125/3125 [==============================] - 1177s 377ms/step - loss: 1.0451 - accuracy: 0.8375
Epoch 2/10
3125/3125 [==============================] - ETA: 0s - loss: 0.4695 - accuracy: 0.9119
Epoch 00002: saving model to ./logs/transformer_ep-02_loss-0.47_acc-0.91.ckpt
3125/3125 [==============================] - 1166s 373ms/step - loss: 0.4695 - accuracy: 0.9119
Epoch 3/10
...
31/32 [============================>.] - ETA: 0s - loss: 0.4831 - accuracy: 0.9023
Epoch 00007: saving model to ./logs/transformer_ep-07_loss-0.48_acc-0.90.ckpt
32/32 [==============================] - 2s 67ms/step - loss: 0.4830 - accuracy: 0.9023
Epoch 8/10
31/32 [============================>.] - ETA: 0s - loss: 0.4402 - accuracy: 0.9108
Epoch 00008: saving model to ./logs/transformer_ep-08_loss-0.44_acc-0.91.ckpt
32/32 [==============================] - 2s 59ms/step - loss: 0.4399 - accuracy: 0.9108
Epoch 9/10
31/32 [============================>.] - ETA: 0s - loss: 0.4137 - accuracy: 0.9146
Epoch 00009: saving model to ./logs/transformer_ep-09_loss-0.41_acc-0.91.ckpt
32/32 [==============================] - 2s 60ms/step - loss: 0.4137 - accuracy: 0.9144
Epoch 10/10
31/32 [============================>.] - ETA: 0s - loss: 0.3943 - accuracy: 0.9168
Epoch 00010: saving model to ./logs/transformer_ep-10_loss-0.39_acc-0.92.ckpt
32/32 [==============================] - 2s 69ms/step - loss: 0.3942 - accuracy: 0.9169
```


Predict

```
import tensorflow as tf
import numpy as np
import os.path

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
```
