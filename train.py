from utils.make_translate import make_translate
import tensorflow as tf
import numpy as np

from dataset import get_dataset, prepare_dataset
from model import get_model


dataset = get_dataset("fr-en")

print("Dataset loaded. Length:", len(dataset), "lines")

train_dataset = dataset[0:10000]

print("Train data loaded. Length:", len(train_dataset), "lines")

(encoder_input,
 decoder_input,
 decoder_output,
 encoder_vocab,
 decoder_vocab,
 encoder_inverted_vocab,
 decoder_inverted_vocab) = prepare_dataset(
    train_dataset,
    shuffle=False,
    lowercase=True,
    max_window_size=20
)

transformer_model = get_model(
    EMBEDDING_SIZE=64,
    ENCODER_VOCAB_SIZE=len(encoder_vocab),
    DECODER_VOCAB_SIZE=len(decoder_vocab),
    ENCODER_LAYERS=2,
    DECODER_LAYERS=2,
    NUMBER_HEADS=4,
    DENSE_LAYER_SIZE=128
)

transformer_model.compile(
    optimizer="adam",
    loss=[
        "sparse_categorical_crossentropy"
    ],
    metrics=[
        "accuracy"
    ]
)

transformer_model.summary()

x = [np.array(encoder_input), np.array(decoder_input)]
print(x)
y = np.array(decoder_output)

name = "transformer"
checkpoint_filepath = "./logs/transformer_ep-{epoch:02d}_loss-{loss:.2f}_acc-{accuracy:.2f}.ckpt"

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs/{}".format(name)
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    verbose=True
)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        translate = make_translate(
            transformer_model, encoder_vocab, decoder_vocab, decoder_inverted_vocab)

        translate("c'est une belle journée.")
        # translate("j'aime manger du gâteau.")
        # translate("c'est une bonne chose.")
        # translate("il faut faire à manger pour nourrir les gens.")
        # translate("tom a acheté un nouveau vélo.")
        # translate("nous faisons du vélo pour aller en ville.")
        # translate("la pêche est bonne aujourd'hui.")
        # translate("mais demain est un autre jour.")


transformer_model.fit(
    x,
    y,
    epochs=10,
    batch_size=32,
    callbacks=[
        model_checkpoint_callback,
        tensorboard_callback,
        CustomCallback()
    ]
)
