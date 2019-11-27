import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# np.random.seed(123)
input_data = np.load('./features-200.npy')
output_data = np.load('./labels-200.npy')
samples = input_data.shape[0]
size = 9
input_shape = (size, size, 1)

input_data = input_data.reshape(samples, size, size, 1)

train_samples = int(0.9 * samples)
input_train, input_test = input_data[:
                                     train_samples], input_data[train_samples:]
output_train, output_test = output_data[:
                                        train_samples], output_data[train_samples:]
# end::mcts_go_cnn_preprocessing[]

# tag::mcts_go_cnn_model[]
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 padding="same",
                 input_shape=input_shape))
model.add(Dropout(rate=0.3))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(500, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(size * size, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# end::mcts_go_cnn_model[]

# tag::mcts_go_cnn_eval[]
model.fit(input_train, output_train,
          batch_size=64,
          epochs=100,
          verbose=1,
          validation_data=(input_test, output_test))
score = model.evaluate(input_test, output_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# end::mcts_go_cnn_eval[]
