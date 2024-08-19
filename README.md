# Movie Dialogue Chatbot

This project implements a chatbot trained on movie dialogues (from the open source [Cornell Movie Dialogues Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)) using a sequence-to-sequence model with LSTM layers. The chatbot can engage in conversations based on patterns learned from movie scripts.

## Features

- Data preprocessing of movie dialogues
- Sequence-to-sequence model with LSTM layers
- Training on a corpus of movie dialogues
- Interactive chat interface

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas

## Installation

1. Download the dataset:
    Download the [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

2. Clone this repository:
```bash
git clone https://github.com/Zairi-Maissene/movie-dialogue-chatbot.git
cd movie-dialogue-chatbot
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data:
- Place your `utterances.jsonl` file in the `data` directory

2. Run the main script:
```bash
python main.py
```
3. Start chatting with the bot when prompted

## Project Structure

- `data_preprocessing.py`: Functions for loading and preprocessing dialogue data
- `model.py`: Definition of the sequence-to-sequence model
- `chatbot.py`: Implementation of the chat interface
- `utils.py`: Utility functions for data preparation
- `main.py`: Main script to run the entire pipeline

## How It Works

1. **Data Preprocessing**: The script reads the movie dialogues from the JSONL file, extracts conversation pairs, and builds a vocabulary.

2. **Model Architecture**: The chatbot uses a sequence-to-sequence model with an encoder-decoder architecture. Both the encoder and decoder use LSTM layers.

3. **Training**: The model is trained on the preprocessed dialogue pairs, learning to generate appropriate responses to given inputs.

4. **Inference**: During chat, the encoder processes the input sentence, and the decoder generates a response based on the encoded input.

## Customization

You can customize various aspects of the chatbot:

- Adjust the `MAX_LENGTH` parameter in `main.py` to change the maximum sentence length.
- Modify the model architecture in `model.py` to experiment with different layer sizes or types.
- Change the `MIN_COUNT` parameter in `data_preprocessing.py` to adjust the vocabulary size.
- Change the pair limits in `main.py` to train on a subset of the data.

## Acknowledgments

- This project was inspired by various seq2seq chatbot implementations.
- Thanks to the open-source community for providing valuable resources and libraries.
