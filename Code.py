from ltp import LTP
import stanza
import torch
import pandas as pd
import re
from collections import Counter

# Load Excel file
time = 'Past'
data = pd.read_excel(f'TSIT_Text.xlsx')

# Add delta smoothing
delta = 0.5

# Initialize LTP and Stanza
ltp = LTP()
nlp = stanza.Pipeline(lang='zh-hans', tokenize_pretokenized=True, download_method=None)

# Move model to GPU if available
if torch.cuda.is_available():
    ltp.to("cuda")

# Define a function to load words from a TXT file
def load_words_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Strip newline characters and return a list
        words = [line.strip() for line in file.readlines()]
    return words

# Load positive and negative words from text files
positive_words = load_words_from_file('正面词_简体.txt')
negative_words = load_words_from_file('负面词_简体.txt')

# Add custom words to the dictionary
ltp.add_words(["西南大学", "被试"], freq=2)

# Initialize list to store extracted features
features = []
participant_num = 0

# Define a function to clean sentences
def clean_sentences(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        # Remove invisible characters and strip whitespace
        cleaned_sentence = re.sub(r'[^\w一-龥，。？！、；：“”‘’（）《》【】]', '', sentence).strip()
        if cleaned_sentence:  # Ensure the sentence is not empty
            cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences

# Define a function to split long text into segments
def split_text(text, max_length=300):
    segments = []
    start = 0
    while start < len(text):
        # Find the last punctuation mark within max_length
        end = start + max_length
        if end < len(text):
            segment_end = max(text.rfind('。', start, end),
                              text.rfind('？', start, end),
                              text.rfind('！', start, end),
                              text.rfind('；', start, end),
                              text.rfind('：', start, end))
            if segment_end == -1: # If no punctuation is found, truncate directly
                segment_end = end
            else:
                segment_end += 1  # Include punctuation mark
        else:
            segment_end = len(text)  # Last segment

        segments.append(text[start:segment_end])
        start = segment_end
    return segments


# Process data row by row
for index, row in data.iterrows():
    text = row[time]
    number = row['ID']
    print(participant_num+1)

    # Split text into segments
    text_segments = split_text(text, max_length=300)
    tokens, pos_tags, dep_relations = [], [], []

    # Process each segment with LTP
    for segment in text_segments:
        try:
            # Perform word segmentation, POS tagging, and dependency parsing
            pipeline_segment = ltp.pipeline([segment], tasks=['cws', 'pos', 'dep'])
            tokens.extend(pipeline_segment.cws[0])
            pos_tags.extend(pipeline_segment.pos[0])
            dep_relations.extend(pipeline_segment.dep[0])
        except RuntimeError as e:
            print(f"Error processing segment: {segment}\nError: {e}")

    # Sentence splitting and cleaning
    sentences = re.split(r'[。？！；：]', text)
    cleaned_sentences = clean_sentences(sentences)
    print(sentences)

    # Dependency parsing for each cleaned sentence
    segments = []
    dep_relations = []
    for sentence in cleaned_sentences:
        try:
            pipeline_sentence = ltp.pipeline([sentence], tasks=['cws', 'pos', 'dep'])
            segments.append(pipeline_sentence.cws[0])
            dep_relations.append(pipeline_sentence.dep[0])
        except RuntimeError as e:
            print(f"Error processing sentence: {sentence}\nError: {e}")

    # Initialize feature dictionary
    feature_dict = {'number': number}

    # Remove punctuation before lexical statistics
    tokens_no_punctuation = [token for token in tokens if re.match(r'\w', token)]
    print(tokens_no_punctuation)

    # Use Counter to avoid redundant calculations
    word_counter = Counter(tokens_no_punctuation)

    # 1. Total word count
    total_word_count = len(tokens_no_punctuation)
    feature_dict[f'1_Total_Words_{time}'] = total_word_count

    # 2. Lexical level: positive/negative word ratio
    positive_count = sum([word_counter[word] for word in positive_words])
    negative_count = sum([word_counter[word] for word in negative_words])
    ratio_pos_neg = (positive_count + delta) / (negative_count + delta)
    feature_dict[f'2_Positive/Negative_Words_{time}'] = ratio_pos_neg

    sentences_no_punctuation_for_syntax = [re.sub(r'[^\w一-鿿]', '', ' '.join(segment)) for segment in segments if
                                           segment]
    segment_result = ltp.pipeline(sentences_no_punctuation_for_syntax, tasks=['cws'])
    SEG = segment_result.cws

    # 3. Total sentence count
    total_sentences = len(sentences_no_punctuation_for_syntax)
    feature_dict[f'3_Sentence_Count_{time}'] = total_sentences

    # 4. Syntactic level: average sentence length
    average_words = total_word_count / total_sentences
    feature_dict[f'4_Avg_Sentence_Length_{time}'] = average_words

    # 5. Semantic level: positive/negative sentence ratio
    def classify_sentence_sentiment(sentence):
        doc = nlp(sentence)
        sentiment_score = doc.sentences[0].sentiment
        if sentiment_score == 2:
            return 'positive'
        elif sentiment_score == 1:
            return 'neutral'
        else:
            return 'negative'


    sentiment_classification = [classify_sentence_sentiment(' '.join(words)) for words in segments]
    ratio_sentence = (sentiment_classification.count('positive') + delta) / (
                sentiment_classification.count('negative') + delta)
    feature_dict[f'5_Positive/Negative_Sentences_{time}'] = ratio_sentence

# Append features to the list
features.append(feature_dict)
participant_num += 1

# Convert all features to a DataFrame
features_df = pd.DataFrame(features)

# Save results to a new Excel file
features_df.to_excel(f'text_feature_{time}.xlsx', index=False)
