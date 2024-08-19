# main.py

import os
from data_preprocessing import prepare_data
from model import create_model, create_inference_models
from utils import prepare_data_for_model
from chatbot import chat

const PAIR_LIMITS = 10000
def main():
    corpus_name = "movie-corpus"
    corpus_file = os.path.join("data", "utterances.jsonl")
    max_length = 10

    voc, pairs = prepare_data(corpus_name, corpus_file, max_length)
    
    # Limit pairs for testing
    pairs = pairs[:PAIR_LIMITS]

    encoder_input_data, decoder_input_data, decoder_target_data = prepare_data_for_model(pairs, voc, max_length)

    model = create_model(voc.num_words, max_length)
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=64, epochs=1, validation_split=0.2)

    model.save('chatbot_model.h5')

    encoder_model, decoder_model = create_inference_models(model, max_length)

    chat(encoder_model, decoder_model, voc, max_length)

if __name__ == "__main__":
    main()