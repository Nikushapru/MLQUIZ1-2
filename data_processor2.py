import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split

df = pd.read_csv("../Darknet.csv")

df = df.drop(["Flow ID", "Timestamp", "Label2", "Src IP", "Dst IP"], axis=1)

df = df.dropna()

label_encoder1 = LabelEncoder()
df['Label1'] = label_encoder1.fit_transform(df['Label1'])

df.to_csv("processed.csv", index=False)

df = pd.read_csv("processed.csv")

features = df.drop(['Label1'], axis=1)
label = df['Label1']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  