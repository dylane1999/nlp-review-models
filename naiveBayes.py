# get the list of files from the given directory
def get_files_from_dir(self, directory: str):
    return [f for f in listdir(directory) if isfile(join(directory, f))]


# perform the needed text processing on the given text and export as a tokenized array
def process_string(self, text: str) -> str:
    englishStopwords = stopwords.words("english")  # non-neccesary words
    text = text.lower()  # case folding
    # remove punctuation
    text = "".join([char for char in text if char not in self.punctuation])
    words = word_tokenize(text)
    removed = [word for word in words if word not in englishStopwords]
    stemmed = [self.porter.stem(word) for word in removed]
    return stemmed


def tokenize_files(self, files: list, dir: str) -> list:
    cleaned_positive_files = []
    for file in files:
        file_path = str.format("{}/{}", dir, file)
        with open(file_path) as f:
            raw_text = f.read()
            cleaned_positive_files.append(self.process_string(raw_text))
    return cleaned_positive_files


def get_raw_text_from_files(self, files: list, dir: str) -> list:
    raw_text = []
    for file in files:
        file_path = str.format("{}/{}", dir, file)
        with open(file_path) as f:
            file_text_in_lines = f.read()
            raw_text.append(file_text_in_lines)
    return raw_text


def get_word_occurrences(self, tokenized_files: list):
    word_occurrences = {}
    word_occurrences["positive"] = 0
    word_occurrences["negative"] = 0
    total_num_words = 0
    for file in tokenized_files:
        # calc number exclams
        # calc number pos/neg/words
        for word in file:
            if self.is_word_positive(word):
                word_occurrences["positive"] += 1
            if self.is_word_negative(word):
                word_occurrences["negative"] += 1
            if word not in word_occurrences:
                word_occurrences[word] = 0
            word_occurrences[word] += 1
            total_num_words += 1
    return word_occurrences, total_num_words


def get_word_probability(self, word_occurrences: dict, total_num_words: int) -> dict:
    word_probabilities = {}
    for word, num_occurrences in word_occurrences.items():
        word_probabilities[word] = num_occurrences / total_num_words
    return word_probabilities
