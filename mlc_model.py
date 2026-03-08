import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# load dataset
df = pd.read_csv('data_all_cancer.csv')

df = df.transpose().reset_index()
df = df.rename(columns={'index': 'SampleID'})


X_df = df.drop(columns=['SampleID', 'class_id'])
y = df['class_id'].astype(int).values

X = X_df.astype(np.int32).values

# dataset stats
n_samples = X.shape[0]
n_features = X.shape[1]
n_classes = len(np.unique(y))
n_categories = int(X.max()) + 1
latent_dim = 100

print("Number of samples:", n_samples)
print("Number of features:", n_features)
print("Number of classes:", n_classes)
print("Number of reconstruction categories:", n_categories)

# one-hot labels
y_cat = to_categorical(y, num_classes=n_classes)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train X shape:", X_train.shape)
print("Test X shape:", X_test.shape)
print("Train y shape:", y_train.shape)
print("Test y shape:", y_test.shape)

inp = Input(shape=(n_features,), name='input_layer')

x = BatchNormalization(name='input_bn')(inp)

# encoder
x = Dense(1024, activation='relu', name='enc_dense_1024')(x)
x = BatchNormalization(name='enc_bn_1024')(x)
x = Dropout(0.3, name='enc_drop_1024')(x)

x = Dense(512, activation='relu', name='enc_dense_512')(x)
x = BatchNormalization(name='enc_bn_512')(x)
x = Dropout(0.3, name='enc_drop_512')(x)

# latent representation
latent = Dense(latent_dim, activation='relu', name='latent_vector')(x)

# classification head
class_output = Dense(n_classes, activation='softmax', name='class_output')(latent)

# decoder
d = Dense(512, activation='relu', name='dec_dense_512')(latent)
d = BatchNormalization(name='dec_bn_512')(d)
d = Dropout(0.3, name='dec_drop_512')(d)

d = Dense(1024, activation='relu', name='dec_dense_1024')(d)
d = BatchNormalization(name='dec_bn_1024')(d)
d = Dropout(0.3, name='dec_drop_1024')(d)

d = Dense(4096, activation='relu', name='dec_dense_4096')(d)
d = BatchNormalization(name='dec_bn_4096')(d)
d = Dropout(0.3, name='dec_drop_4096')(d)

# reconstruction layer
d = Dense(n_features * n_categories, name='decoder_logits')(d)
recon_output = Reshape((n_features, n_categories), name='recon_output')(d)

# build model
model = Model(inputs=inp, outputs=[class_output, recon_output])

model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss={
        'class_output': 'categorical_crossentropy',
        'recon_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    },
    metrics={
        'class_output': ['accuracy']
    },
    loss_weights={
        'class_output': 1.0,
        'recon_output': 1.0
    }
)

# train
history = model.fit(
    X_train,
    {
        'class_output': y_train,
        'recon_output': X_train
    },
    validation_split=0.2,
    epochs=500,
    batch_size=16,
    verbose=1
)

results = model.evaluate(
    X_test,
    {
        'class_output': y_test,
        'recon_output': X_test
    },
    verbose=1
)

print("Evaluation results:", results)

class_pred, recon_pred = model.predict(X_test)

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(class_pred, axis=1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

encoder = Model(inputs=inp, outputs=latent)

# generate latent embeddings
encoded_X = encoder.predict(X)

print("Encoded X shape:", encoded_X.shape)
