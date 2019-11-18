import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# np.random.seed(123)
input_data = np.load('./features-200.npy')
output_data = np.load('./labels-200.npy')
samples = input_data.shape[0]
board_size = 9
input_size = (samples, board_size, board_size, 1)


input_data = input_data.reshape(samples, board_size, board_size, 1)

print("Shapes")
print(input_data.shape)
print(output_data.shape)

train_num = int(0.9 * samples)

# input_train, input_test = input_data[:train_num], input_data[train_num:]
# output_train, output_test = output_data[:train_num], output_data[train_num:]

# model = Sequential()
# model.add(Dense(1000, activation="sigmoid", input_shape=(board_size,)))
# model.add(Dense(500, activation="sigmoid"))
# model.add(Dense(board_size, activation="sigmoid"))

# model.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])
# model.fit(
#     x=input_train,
#     y=output_train,
#     batch_size=81,
#     epochs=16,
#     verbose=1,
#     validation_data=(input_test, output_test))
# model.summary()
# score = model.evaluate(input_test, output_test, verbose=0)
# print("Test Loss: ", score[0])
# print("Test Accuracy: ", score[1])
