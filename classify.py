import csv
import zipfile
from datasets import load_dataset
import numpy as np
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras.regularizers import l2, l1, l1_l2


# Taking an unweighted average of the scores for 6 types of toxicity
# but just return 1 if any of the scores is 1 (so sum is more than 0)
# We may want to come back to this and provide weights for the different
# types of toxicity at a later date
def score(toxic, severe_toxic, obscene, threat, insult, identity_hate):
  score_sum = toxic + severe_toxic + obscene + threat + insult + identity_hate
  # avg = score_sum / 6
  if score_sum > 0:
    return 1
  else:
    return 0

zip_ref = zipfile.ZipFile('glove.6B.50d.txt.zip', 'r')
zip_ref.extractall('glove')
zip_ref.close()

glove_file = 'glove/glove.6B.50d.txt'

dataset = load_dataset('jigsaw_toxicity_pred', data_dir='.')


# Dataset is already split for us, just assigning to test and train variables
print("Splitting dataset...")
train_dataset, test_dataset = dataset['train'], dataset['test']

# Unfortunately dataset['test'] doesn't have the labels so we need to get
# them from our locally downloaded csvs.  But dataset['train'] is ok.
X_train = [comment['comment_text'] for comment in train_dataset]

# Get the test data from our locally downloaded csvs
X_test = []
Y_test = []
total_test = 0

with open('test.csv', 'r', encoding='latin-1') as test_file, \
     open('test_labels.csv', 'r', encoding='latin-1') as labels_file:
  test_reader = csv.reader(test_file, quotechar='"')
  labels_reader = csv.reader(labels_file)
  next(test_reader) # throw out header rows
  next(labels_reader)

  for comment_row, label_row in zip(test_reader, labels_reader):
    total_test += 1
    scores = label_row[1:] #throw out the id, just want the 6 toxicity booleans
    scores = list(map(int, scores))  # change to ints so we can average them

    # this dataset is from a competition that had some data with no scores
    # (labeled with -1s) in the test labels.  We don't want that, so throw
    # those scores out along with the comment
    if -1 in scores:
      continue

    # Otherwise grab the score and comments for our data
    Y_test.append(score(*scores))
    comment = comment_row[1]
    X_test.append(comment)  # throw out the ids


print ("Read in ", total_test, " test comments")
print("Kept ",len(X_test), " test comments")

# Create labels of 0 or 1
print("Calculating overall toxicity scores...")
Y_train = [
  score(comment['toxic'], comment['severe_toxic'], comment['obscene'], comment['threat'], comment['insult'],
    comment['identity_hate'])
  for comment in train_dataset
]

# Load the English stop words from NLTK
stop_words = set(stopwords.words('english'))

# Preprocess the text to remove stop words
X_train_no_stopwords = [' '.join([word.lower() for word in comment.split(' ') if word.lower() not in stop_words]) for comment in X_train]
X_test_no_stopwords = [' '.join([word.lower() for word in comment.split(' ') if word.lower() not in stop_words]) for comment in X_test]

# Tokenize the text data and pad sequences
# keep 5000 words
max_words = 3000
max_sequence_length = 30

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_no_stopwords)

X_train = tokenizer.texts_to_sequences(X_train_no_stopwords)
X_test = tokenizer.texts_to_sequences(X_test_no_stopwords)

X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)
print("Training and test sets tokenized and padded.")

# Zip comments and labels together for a second so we can shuffle them, then unzip them apart
combo_comments_labels = list(zip(X_train, Y_train))
np.random.shuffle(combo_comments_labels)
X_train, Y_train = zip(*combo_comments_labels)

# Tensorflow requires numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

embedding_dim = 50
vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros((vocab_size, embedding_dim))

embeddings = {}
with open(glove_file, 'r', encoding='utf-8') as file:
  for line in file:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embeddings[word] = vector

for word, i in tokenizer.word_index.items():
  embedding_vector = embeddings.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)
model = keras.Sequential([
  embedding_layer,
  keras.layers.LSTM(256, return_sequences=False),
  keras.layers.Dense(50,activation='relu', kernel_regularizer=l2(0.001)), # Hidden layer with 200 nodes and ReLU activation
  keras.layers.Dropout(0.1), # prevent overfitting to training data
  keras.layers.Dense(200,activation='relu', kernel_regularizer=l2(0.001)), # Hidden layer with 200 nodes and ReLU activation
  keras.layers.Dropout(0.2), # prevent overfitting to training data
  keras.layers.Dense(50, activation='relu', kernel_regularizer=l1(0.001)),  # Hidden layer with 200 nodes and ReLU activation
  keras.layers.Dropout(0.1), # prevent overfitting to training data
  keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary sentiment classification
])

# Compile the model
# adam is the most common optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model compiled.")

# Train the model
epochs = 4
batch_size = 128

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
print("Modeled trained.")

# Evaluate the model on the test data
print("Running trained model on test set.")
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

correct_tags = ["toxic" if label == 1 else "nontoxic" for label in Y_test]
test_tags = list(model.predict(X_test))
test_tags = ["toxic" if predicted >= 0.5 else "nontoxic" for predicted in test_tags]

mtrx = nltk.ConfusionMatrix(correct_tags, test_tags)
print()
print(mtrx)
print()
print(mtrx.evaluate())