# chatbot.py

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_response(input_text, encoder_model, decoder_model, voc, max_length):
    input_seq = [voc.word2index.get(word, voc.word2index['OUT']) for word in input_text.split()]
    input_seq = pad_sequences([input_seq], maxlen=max_length, padding='post')

    # Encode the input sequence to get the internal states
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))
    # Populate the first token of target sequence with the start token
    target_seq[0, 0] = voc.word2index['SOS']

    # If the encoder model returns a single state, duplicate it
    if not isinstance(states_value, list):
        states_value = [states_value, states_value]

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        # Predict next token using the decoder model
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = voc.index2word.get(sampled_token_index, 'OUT')

        if sampled_word != 'EOS':
            decoded_sentence += ' ' + sampled_word

        # Exit condition: either hit max length or find stop token
        if sampled_word == 'EOS' or len(decoded_sentence.split()) > max_length - 1:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence.strip()

def chat(encoder_model, decoder_model, voc, max_length):
    print("Start chatting (type 'q' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'q':
            break
        response = generate_response(user_input, encoder_model, decoder_model, voc, max_length)
        print("Chatbot:", response)