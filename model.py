# model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

def create_model(vocab_size, max_length):
    # Encoder
    encoder_inputs = Input(shape=(max_length,))
    encoder_embedding = Embedding(vocab_size, 50, input_length=max_length, trainable=True, name='embedding')(encoder_inputs)
    encoder_lstm = LSTM(400, return_state=True, name='lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_length,))
    decoder_embedding = Embedding(vocab_size, 50, input_length=max_length, trainable=True, name='embedding_1')(decoder_inputs)
    decoder_lstm = LSTM(400, return_sequences=True, return_state=True, name='lstm_1')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax', name='dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_inference_models(model, max_length):
    # Encoder inference model
    encoder_inputs = model.input[0]
    encoder_outputs = model.get_layer('lstm').output
    encoder_states = encoder_outputs[1:]  # Get only the states
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder inference model
    decoder_inputs = Input(shape=(1,))
    decoder_state_input_h = Input(shape=(400,))
    decoder_state_input_c = Input(shape=(400,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding = model.get_layer('embedding_1')
    decoder_lstm = model.get_layer('lstm_1')
    decoder_dense = model.get_layer('dense')

    decoder_embedded = decoder_embedding(decoder_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedded, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model
