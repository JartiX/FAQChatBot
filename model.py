from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import tensorflow as tf
import json


from ENV_VARS import MODEL_PATH, DATA_PATH, RUBERT_TOKENIZER_PATH, RUBERT_MODEL_PATH
from preprocess_text import preprocess_text


with open(DATA_PATH, "r", encoding="UTF-8") as f:
    data = json.load(f)

bert_tokenizer = AutoTokenizer.from_pretrained(RUBERT_TOKENIZER_PATH)
bert_model = TFAutoModel.from_pretrained(RUBERT_MODEL_PATH)

dataset = []
for q_a in data:
    question_embedding = preprocess_text([q_a["question"]], bert_model, bert_tokenizer)
    answer_embedding = preprocess_text([q_a["answer"]], bert_model, bert_tokenizer)
    dataset.append([question_embedding[0], answer_embedding[0]])

dataset = np.array(dataset)

X, Y = [], []
for i in range(dataset.shape[0]):
    for j in range(dataset.shape[0]):
        X.append(np.concatenate([dataset[i, 0, :], dataset[j, 1, :]], axis=0))
        Y.append(1 if i == j else 0)

X = np.array(X)
Y = np.array(Y)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(X.shape[1],)))
model.add(tf.keras.layers.Dense(100, activation='selu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(curve='pr', name='auc')])
model.fit(X, Y, epochs=1000, class_weight={0: 1, 1: np.sqrt(Y.shape[0])-1})
model.save(MODEL_PATH)
