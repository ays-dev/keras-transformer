def decode_with_vocab(sequences, vocab):
  decoded_sequences = []
  decoded_sequence = []

  for sequence in sequences:
    for token in sequence:
      decoded_sequence.append(vocab[token])
    decoded_sequences.append(decoded_sequence)
    encoded_sequence = []

  return decoded_sequences
