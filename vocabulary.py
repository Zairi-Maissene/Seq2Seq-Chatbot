class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "OUT": 3}
        self.word2count = {"PAD": 1, "SOS": 1, "EOS": 1, "OUT": 1}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "OUT"}
        self.num_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = [k for k, v in self.word2count.items() if v >= min_count or k in ["PAD", "SOS", "EOS", "OUT"]]

        print(f'keep_words {len(keep_words)} / {len(self.word2index)} = {len(keep_words) / len(self.word2index):.4f}')

        # Reset dictionaries
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "OUT": 3}
        self.word2count = {"PAD": 1, "SOS": 1, "EOS": 1, "OUT": 1}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "OUT"}
        self.num_words = 4

        for word in keep_words:
            if word not in self.word2index:  # Don't re-add special tokens
                self.add_word(word)