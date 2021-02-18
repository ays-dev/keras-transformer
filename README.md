
################################################
#                                              #
#              Keras Transformer               #
#                                              #
################################################
# from keras_transformer import get_model, decode

# model = get_model(
#     token_num = max(len(encoder_vocab), len(decoder_vocab)),
#     embed_dim = 32,
#     encoder_num = 1,
#     decoder_num = 1,
#     head_num = 1,
#     hidden_dim = 128,
#     # dropout_rate = 0.05,
#     use_same_embed = False,
# )

# model.compile("adam", "sparse_categorical_crossentropy")
# model.summary()

# x = [np.array(encoder_input), np.array(decoder_input)]
# y = np.array(decoder_output)

# model.fit(x, y, epochs=5, batch_size=32)

# def translate(sentence):
#   sentence_tokens = [tokens + ["<stop>", "<pad>"] for tokens in [sentence.split(" ")]]
#   tr_input = [list(map(lambda x: encoder_vocab[x], tokens)) for tokens in sentence_tokens][0]
#   decoded = decode(
#       model, 
#       tr_input, 
#       start_token = decoder_vocab["<start>"],
#       end_token = decoder_vocab["<stop>"],
#       pad_token = decoder_vocab["<pad>"]
#   )

#   print("Original: {}".format(sentence))
#   print("Trad: {}".format(" ".join(map(lambda x: decoder_inverted_vocab[x], decoded[1:-1]))))

# translate("hello")
# translate("today is a good day")
# translate("how are you ?")
# translate("go run")