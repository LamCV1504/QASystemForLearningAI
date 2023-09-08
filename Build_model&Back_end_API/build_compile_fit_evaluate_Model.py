import datetime
import numpy as np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras import metrics

# thiet lap chung
from config import EMBEDDING_DIM, FILTER_SIZE, NUM_FILTERS, DROP, EPOCH, BATCH_SIZE, L2
from config import get_path_currying
from preprocessing_data import load_data_temp
print(datetime.datetime.now())
t1 = datetime.datetime.now()

model_path = get_path_currying('Model\\')

# model_json_full = "D:\\NCKH_HTHD\\Model\\chuong2.json"
file_name = 'all'
model_json_full = model_path(file_name + '.json');

num_labels = 5
np.random.seed(0)

#load data
X_train, y_train, X_test, y_test, X_val, y_val,  embedding_layer = load_data_temp()
 
#build model
print(X_train.shape)
sequence_length = X_train.shape[1]
inputs = Input(shape=(sequence_length,))
embedding = embedding_layer(inputs)
reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)
 
conv_0 = Conv2D(NUM_FILTERS, (FILTER_SIZE[0], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape) 
conv_1 = Conv2D(NUM_FILTERS, (FILTER_SIZE[1], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_2 = Conv2D(NUM_FILTERS, (FILTER_SIZE[2], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_3 = Conv2D(NUM_FILTERS, (FILTER_SIZE[2], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_4 = Conv2D(NUM_FILTERS, (FILTER_SIZE[2], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)

maxpool_0 = MaxPooling2D((sequence_length - FILTER_SIZE[0] + 1, 1), strides=(1,1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - FILTER_SIZE[1] + 1, 1), strides=(1,1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - FILTER_SIZE[2] + 1, 1), strides=(1,1))(conv_2)
maxpool_3 = MaxPooling2D((sequence_length - FILTER_SIZE[2] + 1, 1), strides=(1,1))(conv_3)
maxpool_4 = MaxPooling2D((sequence_length - FILTER_SIZE[2] + 1, 1), strides=(1,1))(conv_4)

merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4], axis=1)
flatten = Flatten()(merged_tensor)
dropout = Dropout(DROP)(flatten)
output = Dense(units = num_labels, activation='softmax',kernel_regularizer=regularizers.l2(L2))(dropout)
model = Model(inputs, output)
adam = Adam(learning_rate=1e-3)

model.summary()

#compile_model
model.compile(loss = 'categorical_crossentropy',
              optimizer= adam,
              metrics=['acc', metrics.Precision(), metrics.Recall()])

callbacks = [EarlyStopping(monitor='val_loss')]

# fit_model
# checkpoint_filepath = 'D:\\NCKH_HTHD\\Model\\chuong2.h5'
checkpoint_filepath = model_path(file_name + '.h5');

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=False)  

model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,
          validation_data=(X_val, y_val),
          callbacks=[model_checkpoint_callback])  # starts training

#save model
model_json = model.to_json()
with open(model_json_full, 'w') as json_file:
    json_file.write(model_json)

#evaluate model
scores = model.evaluate(X_test, y_test)
print("Loss:", (scores[0]))
print("Accuracy:", (scores[1]*100))
print("Precision:", (scores[2]*100))
print("Recall:", (scores[3]*100))

print(datetime.datetime.now())
t2 = datetime.datetime.now()
print("Time:")
print(t2-t1)
