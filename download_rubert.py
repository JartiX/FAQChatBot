from transformers import AutoTokenizer, TFAutoModel

from ENV_VARS import MODEL_NAME, RUBERT_MODEL_PATH, RUBERT_TOKENIZER_PATH


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModel.from_pretrained(MODEL_NAME, from_pt=True)

tokenizer.save_pretrained(RUBERT_TOKENIZER_PATH)
model.save_pretrained(RUBERT_MODEL_PATH)
