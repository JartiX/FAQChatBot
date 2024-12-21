def preprocess_text(text: list, model, tokenizer):
    return model(**tokenizer(text, return_tensors='tf',
                                       padding=True, truncation=True))['last_hidden_state'][:, 0, :].numpy()
