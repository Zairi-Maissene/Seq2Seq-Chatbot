# data_preprocessing.py
import json
import re
import unicodedata
import os
from typing import Dict, List, Tuple
from vocabulary import Voc

def load_lines_and_conversations(file_name: str) -> Tuple[Dict, Dict]:
    lines = {}
    conversations = {}
    with open(file_name, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line_json = json.loads(line)
            line_obj = {
                "lineID": line_json["id"],
                "characterID": line_json["speaker"],
                "text": line_json["text"]
            }
            lines[line_obj['lineID']] = line_obj

            conv_id = line_json["conversation_id"]
            if conv_id not in conversations:
                conversations[conv_id] = {
                    "conversationID": conv_id,
                    "movieID": line_json["meta"]["movie_id"],
                    "lines": [line_obj]
                }
            else:
                conversations[conv_id]["lines"].insert(0, line_obj)

    return lines, conversations

def extract_sentence_pairs(conversations: Dict) -> List[List[str]]:
    qa_pairs = []
    for conversation in conversations.values():
        for i in range(len(conversation["lines"]) - 1):
            input_line = conversation["lines"][i]["text"].strip()
            target_line = conversation["lines"][i+1]["text"].strip()
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs

def normalize_string(s: str) -> str:
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def filter_pair(p: List[str], max_length: int) -> bool:
    return len(p[0].split()) < max_length and len(p[1].split()) < max_length

def filter_pairs(pairs: List[List[str]], max_length: int) -> List[List[str]]:
    return [pair for pair in pairs if filter_pair(pair, max_length)]

def prepare_data(corpus_name: str, corpus_file: str, max_length: int) -> Tuple[Voc, List[List[str]]]:
    print("Preparing training data...")
    voc = Voc(corpus_name)
    lines, conversations = load_lines_and_conversations(corpus_file)
    pairs = extract_sentence_pairs(conversations)

    print(f"Read {len(pairs)} sentence pairs")
    pairs = filter_pairs(pairs, max_length)
    print(f"Trimmed to {len(pairs)} sentence pairs")

    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])

    print(f"Counted words: {voc.num_words}")
    return voc, pairs
