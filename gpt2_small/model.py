import tensorflow as tf # type: ignore
from gpt2_small.embedding import WTE, WPE
from gpt2_small.gpt_block import GPT2LayerNorm, GPT2Block

def lm_head(x, wte):
    return tf.einsum("btd,vd->btv", x, wte)

class GPT2Model(tf.keras.Model):
  def __init__(self):
    super().__init__(name="gpt2")

    self.wte = WTE(vocab_size=50257, d_model=768)
    self.wpe = WPE(max_position=1024, d_model=768)

    self.blocks = [GPT2Block(d_model=768, n_head=12) for _ in range(12)]

    self.ln_f = GPT2LayerNorm(dim=768)

  def call(self, inputs):
      input_ids = inputs["input_ids"]
      attention_mask = inputs.get("attention_mask", None)

      B = tf.shape(input_ids)[0]
      T = tf.shape(input_ids)[1]

      positions = tf.range(T)[None, :]
      x = self.wte(input_ids) + self.wpe(positions)

      for block in self.blocks:
          x = block(x, attention_mask)

      x = self.ln_f(x)
      logits = lm_head(x, self.wte.embedding.embeddings)
      return logits
