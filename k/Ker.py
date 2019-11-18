import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

(train_input, train_output), (test_input, test_output) = mnist.load_data()


oneHotDim = 784
print("x test train")
train_input = train_input.reshape(train_input.shape[0], oneHotDim) \
    .astype('float32')
test_input = test_input.reshape(test_input.shape[0], oneHotDim) \
    .astype('float32')

train_input /= 255
test_input /= 255


train_output = keras.utils.to_categorical(train_output, 10)
test_output = keras.utils.to_categorical(test_output, 10)

model = Sequential()
model.add(Dense(392, activation='relu', input_shape=(oneHotDim,)))
model.add(Dense(196, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])
model.fit(x=train_input, y=train_output, batch_size=50, epochs=25)
score = model.evaluate(test_input, test_output)
print("Test Loss: ", score[0])
print("Test Accuracy: ", score[1])
