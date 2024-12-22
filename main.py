import tkinter as tk
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import json

from ENV_VARS import MODEL_PATH, DATA_PATH, RUBERT_MODEL_PATH, RUBERT_TOKENIZER_PATH
from preprocess_text import preprocess_text

bert_tokenizer = AutoTokenizer.from_pretrained(RUBERT_TOKENIZER_PATH)
bert_model = TFAutoModel.from_pretrained(RUBERT_MODEL_PATH)

with open(DATA_PATH, "r", encoding="UTF-8") as f:
    data = json.load(f)

dataset = []
for q_a in data:
    embedding_question = preprocess_text(
        [q_a["question"]], bert_model, bert_tokenizer)
    embedding_answer = preprocess_text(
        [q_a["answer"]], bert_model, bert_tokenizer)

    dataset.append([embedding_question[0], embedding_answer[0]])

dataset = np.array(dataset)

model = tf.keras.models.load_model(MODEL_PATH)


def show_answer(answer):
    answer_window = tk.Toplevel(window)
    answer_window.title("Ответ")

    answer_label = tk.Label(
        answer_window, text="Ответ:", font=("Helvetica", 14))
    answer_label.pack(padx=20, pady=10)

    answer_text = tk.Label(answer_window, text=answer,
                           wraplength=400, font=("Helvetica", 12))
    answer_text.pack(padx=20, pady=10)

    close_button = tk.Button(answer_window, text="Закрыть",
                             command=answer_window.destroy, font=("Helvetica", 12))
    close_button.pack(pady=10)


def get_answer():
    question = question_entry.get()
    if question == "":
        return
    embedding_question = preprocess_text(
        [question], bert_model, bert_tokenizer)[0]

    p = []
    for i in range(dataset.shape[0]):
        embedding_answer = dataset[i, 1]
        combined_embedding = np.concatenate(
            [embedding_question, embedding_answer])
        prediction = model.predict(np.expand_dims(
            combined_embedding, axis=0), verbose=False)[0, 0]
        p.append([i, prediction])

    p = np.array(p)
    ans = np.argmax(p[:, 1])

    answer = "Ответ: " + data[ans]["answer"]
    show_answer(answer)


window = tk.Tk()
window.title("Вопрос-Ответ")

question_label = tk.Label(
    window, text="Введите ваш вопрос:", font=("Helvetica", 14))
question_label.pack(padx=10, pady=5)

question_entry = tk.Entry(window, width=50, font=("Helvetica", 12))
question_entry.pack(padx=10, pady=5)

submit_button = tk.Button(window, text="Получить ответ", command=get_answer, font=(
    "Helvetica", 12), bg="#4CAF50", fg="white", relief="raised")
submit_button.pack(padx=10, pady=20)

window.mainloop()
