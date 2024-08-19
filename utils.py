# utils.py

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

def prepare_data_for_model(pairs, voc, max_length):
    encoder_input_data = []
    decoder_input_data = []
    decoder_target_data = []

    for input_text, target_text in pairs:
        encoder_input = [voc.word2index.get(word, voc.word2index['OUT']) for word in input_text.split()]
        decoder_input = [voc.word2index['SOS']] + [voc.word2index.get(word, voc.word2index['OUT']) for word in target_text.split()]
        decoder_target = decoder_input[1:] + [voc.word2index['EOS']]

        encoder_input_data.append(encoder_input)
        decoder_input_data.append(decoder_input)
        decoder_target_data.append(decoder_target)

    encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_length, padding='post')
    decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_length, padding='post')
    decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_length, padding='post')
    decoder_target_data = to_categorical(decoder_target_data, num_classes=voc.num_words)

    return encoder_input_data, decoder_input_data, decoder_target_data