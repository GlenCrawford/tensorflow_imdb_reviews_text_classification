import tensorflow as tf
from tensorflow import keras
import numpy as np

### Import and preprocess train and test data ###

# Load the IMDB movie reviews dataset. 50,000 total reviews. Split it up into two subsets, for training and testing, of 25,000 each.
# Each data input is a movie review as an array of "words", represented as integers that map to a word in the index.
# Word index is a dictionary of nearly a hundred thousand words. Keys are the words (string), values are their index (integer).
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
word_index = data.get_word_index()

# Increment the indexes (the dictionary values) by 3 3 to accommodate special characters, then add those.
word_index = { word: (index + 3) for word, index in word_index.items() }

# <PAD> is used to make each review the same length, by appending padding to smaller ones.
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

# Invert the index to swap the keys and values, so the indexes (integer) are the keys, and the words (string) are the values.
inverted_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Truncate reviews to 250 characters, and pad reviews shorter than 250 with <PAD> special characters.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index['<PAD>'], padding = 'post', maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index['<PAD>'], padding = 'post', maxlen = 250)

# Split up the training data to slice out 10,000 inputs for validation data, so we can test the model with data that it hasn't already seen, in order to make sure it doesn't just memorize the training data!
validation_data = train_data[:10000] # 10,000
train_data = train_data[10000:] # 15,000

validation_labels = train_labels[:10000] # 10,000
train_labels = train_labels[10000:] # 15,000

# Convert a plain-text review (an array of words) into corresponding word indexes.
def encode_review(review_as_words):
  encoded = [1] # begin with the start special character
  for word in review_as_words:
    if word.lower() in word_index:
      encoded.append(word_index[word.lower()])
    else:
      encoded.append(2) # USe the <UNK> special character.
  return encoded

# Convert a review (which is an array of the indexes of the words) into the corresponding words.
def decode_review(review_as_indexes):
  return ' '.join([inverted_word_index.get(word_index, '?') for word_index in review_as_indexes])

### Define the architecture of the neural network ###
model = keras.Sequential()

# Embedding layer: Going to try and group similar words together. Will generate a vector for each word and pass them to next layer.
# Each word vector = 16 dimensions/coefficients.
# Initially create 88000 vectors, one for each word. The layer will group the vectors of words based on them sharing similar contexts within the reviews.
# Output is a 16 dimension/coefficient vector for each word. The closer two vectors are to each other, the more similarly they are used in the context of the reviews.
model.add(keras.layers.Embedding(88000, 16))

# Takes the word vectors from the previous layer and averages them out.
model.add(keras.layers.GlobalAveragePooling1D())

# Hidden layer: 16 neurons.
model.add(keras.layers.Dense(16, activation = 'relu'))

# Output layer: 1 neuron with a value between 0 and 1 (squashed using the sigmoid function) denoting whether the review is positive or negative.
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

# Uncomment to output the summary of the model's architecture.
# model.summary()

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

### Do some manual testing using one of the imported reviews ###
# Note: Suspect a bug here: prediction per input has a length of 250 instead of 1?

print("\nManual testing using one of the imported reviews:")
index_of_review_to_test = 0
test_review_prediction = model.predict([test_data[index_of_review_to_test]])
print('Test review: ')
print(decode_review(test_data[index_of_review_to_test]))
print('Prediction: ' + str(test_review_prediction[0]))
print('Actual: ' + str(test_labels[index_of_review_to_test]))

### Do some manual testing using some custom data ###

print("\nManual testing using some custom data:")

review_to_test = "Of all the animation classics from the Walt Disney Company, there is perhaps none that is more celebrated than \"The Lion King.\" Its acclaim is understandable: this is quite simply a glorious work of art. \"The Lion King\" gets off to a fantastic start. The film's opening number, \"The Circle of Life,\" is outstanding. The song lasts for about four minutes, but from the first sound, the audience is floored. Not even National Geographic can capture something this beautiful and dramatic. Not only is this easily the greatest moment in film animation, this is one of the greatest sequences in film history. The story that follows is not as majestic, but the film has to tell a story. Actually, the rest of the film holds up quite well. The story takes place in Africa, where the lions rule. Their king, Mufasa (James Earl Jones) has just been blessed with a son, Simba (Jonathan Taylor Thomas), who goes in front of his uncle Scar (Jeremy Irons) as next in line for the throne. Scar is furious, and sets in motion plans to usurp the throne for himself. After a tragedy occurs and Mufasa is killed, Scar persuades Simba to flee, leaving himself as king. Simba grows up in exile, but he learns that while he can run away from his past, he can never escape it. When viewing the film, it is obvious that \"The Lion King\" is quite different from its predecessors (and successors). This is an epic story that contains more dramatic power than all the other Disney films combined. While there are definitely some light-hearted moments, there is no denying the dark drama that takes up the bulk of the story. While it could be argued that Disney is the champion of family entertainment, this film is not for the very young. Some of the sequences are very dark and violent, many bordering on frightening, even for the older crowd.The voice actors are terrific. Jonathan Taylor Thomas brings a large dose of innocence to Young Simba. He's mischievous, but also terribly naive. His older counterpart, voiced by Matthew Broderick, equals him. He's older, but no less mature. The voices are so similar that it's almost impossible not to believe that they are the same character at different ages. Perhaps no one could have been better suited for the role of Mufasa than James Earl Jones. His baritone voice gives the Mufasa a quality of great power and wisdom; there is no question that his role is king. As Scar, Jeremy Irons is pitch-perfect. The drawing of the character is villainous, but Irons' vocal work complements the animation to create one of the most memorable, and vicious, villains in Disney history. He's unquestionably evil, but he's also clever, which makes him all the more dangerous. Manipulation, not violence is his greatest weapon. Providing some much needed comic relief are Nathan Lane and Ernie Sabella as Timon and Pumbaa, two other outcasts (a meerkat and a warthog), and Rowan Atkinson as Zazu. While there is definite fun from these characters, neither the actors nor the filmmakers allow them to go over-the-top and destroy the mood of the film.Disney's animated features are known for their gorgeous artwork. Nowhere is this more apparent than in \"The Lion King.\" Every single frame is jaw-dropping. The colors are rich, and the drawings are sharp and beautiful. One of the pitfalls of animation (both computer and hand-drawn) is that there is sometimes a visible distance between the subject and the background, making it seem as if the figure animation was cut and pasted on the background (this is obviously what happens, but it is up to the artists to make sure that it isn't noticeable). There is none of that here.Throughout the Golden Age of Disney animation, the films have been musicals. \"The Lion King\" is no different, and the songs are brilliant. All of the numbers are standouts (\"Can You Feel the Love Tonight\" won the Oscar, but in my opinion, \"The Circle of Life\" was better). In the cases of Simba and Nala (Simba's girlfriend), both young and old, there is a noticeable difference between the speaking and singing parts (everyone else does their own singing and speaking), but never mind. It still works, and that's what's important. \"The Lion King\" is not flawless, but on first viewing, they aren't noticeable, and it is likely that the young won't ever notice them. \"Beauty and the Beast\" was the first animated film to get an Oscar nomination for Best Picture (it lost to \"The Silence of the Lambs\"), and is thus far the only animated film to receive such an honor. That being the case, it's hard to understand why \"The Lion King\" was not given the same distinction. The two films are more or less equal in quality, and the nominees for the honor that year were not strong. If you haven't already, see \"The Lion King.\" You won't be disappointed."
review_to_test = review_to_test.replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace(':', '').replace('"', '').strip().split(' ')
review_to_test = encode_review(review_to_test)
review_to_test = keras.preprocessing.sequence.pad_sequences([review_to_test], value = word_index['<PAD>'], padding = 'post', truncating = 'post', maxlen = 250)

test_review_prediction = model.predict(review_to_test)
print('Prediction: ' + str(test_review_prediction[0][0]))
