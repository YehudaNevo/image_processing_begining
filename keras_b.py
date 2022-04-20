import numpy as np
from numpy import genfromtxt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report

data = genfromtxt("DATA/bank_note_data.txt", delimiter=",")

labels = data[:, 4]
features = data[:, 0:4]

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# scale_obj = MinMaxScaler()
# scaled_X_train = scale_obj.fit(X_train)
# scaled_X_test = scale_obj.fit(X_test)


model = Sequential()
model.add(Dense(4, input_dim=4, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])  # that the default param..
model.fit(X_train, y_train, epochs=50, verbose=2)

predictions = (model.predict(X_test) > 0.5).astype("int32")

c_matrix = confusion_matrix(y_test,predictions)
print(c_matrix)
print(classification_report(y_test,predictions))
