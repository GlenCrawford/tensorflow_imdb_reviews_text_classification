import tensorflow as tf
from tensorflow import keras
import numpy as np

### Import and preprocess training and test data ###

# Load the IMDB movie reviews dataset. 50,000 total reviews. Split it up into two subsets, for training and testing, of 25,000 each.
# Each data input is a movie review as an array of "words", represented as integers that map to a word in the index.
# Word index is a dictionary of nearly a hundred thousand words. Keys are the words (string), values are their index (integer).
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
word_index = data.get_word_index()

# Increment the indexes (the dictionary values) by 3 to accommodate special characters, then add those.
word_index = { word: (index + 3) for word, index in word_index.items() }

# <PAD> is used to make each review the same length, by appending padding to smaller ones.
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

# Invert the index to swap the keys and values, so the indexes (integer) are the keys, and the words (string) are the values.
inverted_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Truncate reviews to 250 characters, and pad reviews shorter than 250 with <PAD> special characters.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index['<PAD>'], padding = 'post', truncating = 'post', maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index['<PAD>'], padding = 'post', truncating = 'post', maxlen = 250)

# Split up the training data to slice out 10,000 inputs for validation data, so we can test the model with data that it hasn't already seen, in order to make sure it doesn't just memorize the training data!
validation_data = train_data[:10000] # 10,000
train_data = train_data[10000:] # 15,000

validation_labels = train_labels[:10000] # 10,000
train_labels = train_labels[10000:] # 15,000

# Convert a plain-text review (an array of words) into corresponding word indexes.
def encode_review(review_as_words):
  encoded = [1] # Begin with the start special character
  for word in review_as_words:
    if word.lower() in word_index:
      encoded.append(word_index[word.lower()])
    else:
      encoded.append(2) # Use the <UNK> special character.
  return encoded

# Convert a review (which is an array of the indexes of the words) into the corresponding words.
def decode_review(review_as_indexes):
  return ' '.join([inverted_word_index.get(word_index, '?') for word_index in review_as_indexes])

### Define the architecture of the neural network ###
model = keras.Sequential()

# Embedding layer: Going to try and group similar words together. Will generate a vector for each word and pass them to the next layer.
# Each word vector = 16 dimensions/coefficients.
# Initially create 88,000 vectors, one for each word. The layer will group the vectors of words based on them sharing similar contexts within the reviews.
# Output is a 16 dimension/coefficient vector for each word. The closer two vectors are to each other, the more similarly they are used in the context of the reviews.
model.add(keras.layers.Embedding(88000, 16))

# Takes the word vectors from the previous layer and averages them out.
model.add(keras.layers.GlobalAveragePooling1D())

# Hidden layer: 16 neurons.
model.add(keras.layers.Dense(16, activation = 'relu'))

# Output layer: 1 neuron with a value between 0 and 1 (squashed using the sigmoid function) denoting whether the review is positive or negative.
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

# Output the summary of the model's architecture.
model.summary()

# Loss function will classify the difference between the prediction and the expected value (0 or 1), ie, 0.2 is not too different to 0.
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

### Train the model ###

model.fit(train_data, train_labels, epochs = 40, batch_size = 512, validation_data = (validation_data, validation_labels), verbose = 1)

loss, accuracy = model.evaluate(test_data, test_labels)
print('Training complete! Loss: ' + str(loss) + ', Accuracy: ' + str(accuracy))

# Now it has been trained, save the model.
model.save('model.h5')

# Uncomment to load the saved model, then can skip training if desired.
# model = keras.models.load_model('model.h5')

### Do some manual testing using some imported reviews ###

print("\nManual testing using some imported reviews:")

number_of_reviews_to_test = 10
start_index_of_reviews_to_test = 0
end_index_of_reviews_to_test = start_index_of_reviews_to_test + number_of_reviews_to_test

reviews_to_test = test_data[start_index_of_reviews_to_test:end_index_of_reviews_to_test]
test_reviews_predictions = model.predict(reviews_to_test)

for index, review_to_test in enumerate(reviews_to_test):
  test_data_index = start_index_of_reviews_to_test + index
  print('Review: ' + decode_review(review_to_test))
  print('Prediction: ' + str(test_reviews_predictions[index][0]))
  print('Actual: ' + str(test_labels[test_data_index]))
  print('')

### Do some manual testing using some custom data ###

print("\nManual testing using some custom data:")

# Last three paragraphs of Roger Ebert's review of Forrest Gump. 4 out of 5 starts, lots of positive language, ends with "What a magical movie", so it's a positive review. Prediction should therefore be very close to 1.
review_to_test = "The movie is ingenious in taking Forrest on his tour of recent American history. The director, Robert Zemeckis, is experienced with the magic that special effects can do (his credits include the \"Back To The Future\" movies and \"Who Framed Roger Rabbit\"), and here he uses computerized visual legerdemain to place Gump in historic situations with actual people. Forrest stands next to the schoolhouse door with George Wallace, he teaches Elvis how to swivel his hips, he visits the White House three times, he's on the Dick Cavett show with John Lennon, and in a sequence that will have you rubbing your eyes with its realism, he addresses a Vietnam-era peace rally on the Mall in Washington. Special effects are also used in creating the character of Forrest's Vietnam friend Lt. Dan (Gary Sinise), a Ron Kovic type who quite convincingly loses his legs. Using carefully selected TV clips and dubbed voices, Zemeckis is able to create some hilarious moments, as when LBJ examines the wound in what Forrest describes as \"my butt-ox.\" And the biggest laugh in the movie comes after Nixon inquires where Forrest is staying in Washington, and then recommends the Watergate. (That's not the laugh, just the setup.) As Forrest's life becomes a guided tour of straight-arrow America, Jenny (played by Robin Wright) goes on a parallel tour of the counterculture. She goes to California, of course, and drops out, tunes in, and turns on. She's into psychedelics and flower power, antiwar rallies and love-ins, drugs and needles. Eventually it becomes clear that between them Forrest and Jenny have covered all of the landmarks of our recent cultural history, and the accommodation they arrive at in the end is like a dream of reconciliation for our society. What a magical movie."
review_to_test = review_to_test.replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace(':', '').replace('"', '').strip().split(' ')
review_to_test = encode_review(review_to_test)
review_to_test = keras.preprocessing.sequence.pad_sequences([review_to_test], value = word_index['<PAD>'], padding = 'post', truncating = 'post', maxlen = 250)

test_review_prediction = model.predict(review_to_test)
print('Prediction: ' + str(test_review_prediction[0][0]))
print('Expectation: 1')
