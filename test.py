import numpy as np
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('ep_model.keras')

groups = [
    'Confederal Group of the European United Left - Nordic Green Left',
    'Europe of Freedom and Direct Democracy Group',
    'Europe of Nations and Freedom Group',
    'European Conservatives and Reformists Group',
    'Group of the Alliance of Liberals and Democrats for Europe',
    "Group of the European People's Party (Christian Democrats)",
    'Group of the Greens/European Free Alliance',
    'Group of the Progressive Alliance of Socialists and Democrats in the European Parliament'
]

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

mep_index = 0
mep = df_filtered.iloc[mep_index]
actual_group = mep['EPG']
mep_name = mep['Fname'] + ' ' + mep['Lname']

votes = mep[vote_columns].values.astype('float32').reshape(1, -1)

prediction = model.predict(votes)
predicted_index = np.argmax(prediction)
predicted_group = groups[predicted_index]
confidence = prediction[0][predicted_index] * 100

print(f"MEP: {mep_name}")
print(f"Actual group:    {actual_group}")
print(f"Predicted group: {predicted_group}")
print(f"Confidence: {confidence:.1f}%")
