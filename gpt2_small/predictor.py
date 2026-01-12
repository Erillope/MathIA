import tensorflow as tf
from transformers import GPT2Tokenizer # type: ignore
from .model import GPT2Model
from typing import Tuple

class GPTPredictor():
  def __init__(self, weights_path) -> None:
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model = self.build_model(weights_path)

  def build_model(self, weights_path) -> tf.keras.Model:
    model = GPT2Model()
    dummy_input = tf.zeros((1, 1024), dtype=tf.int32)
    _ = model({"input_ids": dummy_input})
    model.load_weights(weights_path)
    return model

  def sample(self, logits, temperature=1.0):
    logits = logits / temperature
    return tf.random.categorical(logits, 1, dtype=tf.int32)[:, 0]

  def generate(self, inputs, max_new_tokens=30, temperature=1.0):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    for _ in range(max_new_tokens):
        outputs = self.model({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        input_ids, attention_mask = self.generate_next_inputs(outputs, temperature, input_ids, attention_mask)

    return input_ids

  def generate_next_inputs(self, outputs, temperature, prev_input, prev_attn) -> Tuple:
    logits = outputs[:, -1, :]
    next_token = self.sample(logits, temperature)
    input_ids = tf.concat([prev_input, next_token[:, None]], axis=-1)
    attention_mask = tf.concat([prev_attn, tf.ones_like(next_token[:, None])], axis=-1)
    return input_ids, attention_mask

  def tokenize(self, prompt):
    return self.tokenizer(
        prompt,
        return_tensors="tf",
        padding=True,
        truncation=True
    )

  def predict(self, prompt: str, max_tokens=10) -> str:
    input_ids = self.tokenize(prompt)
    output_ids = self.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=1.0
    )
    return self.tokenizer.decode(output_ids[0])