from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

docs = pd.read_excel('EP8_Voted_docs.xlsx')
final_votes = docs[docs['Final \nvote?'] == 1]
final_vote_ids = final_votes['Vote ID'].astype(str).tolist()

df = pd.read_excel('EP8_RCVs_2019_06_25.xlsx')
df = df[df['EPG'] != 'Non-attached Members']

info_columns = ['WebisteEpID', 'Fname', 'Lname',
                'Activ', 'Country', 'Party', 'EPG', 'Start', 'End']
vote_columns = [col for col in df.columns if str(col) in final_vote_ids]
df_filtered = df[info_columns + vote_columns].copy()
df_filtered = df_filtered.dropna(subset=['EPG'])

df_filtered[vote_columns] = df_filtered[vote_columns].replace(0, np.nan)
df_filtered[vote_columns] = df_filtered[vote_columns].fillna(5)
df_filtered[vote_columns] = df_filtered[vote_columns].replace(6, 5)

groups = sorted(df_filtered['EPG'].unique())
group_to_num = {group: i for i, group in enumerate(groups)}

y_numbers = np.array([group_to_num[g] for g in df_filtered['EPG']])
num_classes = len(groups)
y = np.eye(num_classes)[y_numbers].astype('float32')
X = df_filtered[vote_columns].values.astype('float32')

indices = np.random.seed(42)
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]


split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

model = Sequential([
    Dense(64, activation='relu', input_shape=(2072,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(8, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

model.save('ep_model.keras')
print("Model saved!")
