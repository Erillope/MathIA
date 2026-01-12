import tensorflow as tf

class GPT2LayerNorm(tf.keras.layers.Layer):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = self.add_weight(name="gamma", shape=(dim,), initializer="ones", trainable=True)
        self.beta  = self.add_weight(name="beta",  shape=(dim,), initializer="zeros", trainable=True)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var  = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) / tf.sqrt(var + self.eps) * self.gamma + self.beta


def causal_mask(seq_len):
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    mask = (1.0 - mask) * -1e4
    return mask[None, None, :, :]


class GPT2Attention(tf.keras.layers.Layer):
    def __init__(self, d_model=768, n_head=12):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.c_attn = tf.keras.layers.Dense(
            3 * d_model,
            use_bias=True
        )

        self.c_proj = tf.keras.layers.Dense(
            d_model,
            use_bias=True
        )

    def split_heads(self, x):
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], x.shape[2]
        x = tf.reshape(x, (B, T, self.n_head, self.head_dim))
        return tf.transpose(x, [0, 2, 1, 3])  # (B, nh, T, hs)

    def call(self, x, attention_mask=None):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        qkv = self.c_attn(x)
        q, k, v = tf.split(qkv, 3, axis=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scale = 1.0 / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        att = tf.matmul(q, k, transpose_b=True) * scale

        causal = causal_mask(T)
        causal = tf.reshape(causal, (1, 1, T, T))

        att = att + causal

        if attention_mask is not None:
            padding = tf.cast(attention_mask[:, None, None, :], tf.float32)
            padding = (1.0 - padding) * -1e9
            att = att + padding

        att = tf.nn.softmax(att, axis=-1)

        out = tf.matmul(att, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (tf.shape(out)[0], tf.shape(out)[1], -1))

        return self.c_proj(out)


class GPT2MLP(tf.keras.layers.Layer):
    def __init__(self, d_model=768):
        super().__init__()
        self.c_fc = tf.keras.layers.Dense(4 * d_model)
        self.c_proj = tf.keras.layers.Dense(d_model)

    def gelu(self, x):
        return 0.5 * x * (1 + tf.tanh(
            tf.sqrt(2 / tf.constant(3.141592653589793)) *
            (x + 0.044715 * tf.pow(x, 3))
        ))

    def call(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class GPT2Block(tf.keras.layers.Layer):
    def __init__(self, d_model=768, n_head=12):
        super().__init__()
        self.ln_1 = GPT2LayerNorm(d_model)
        self.attn = GPT2Attention(d_model, n_head)
        self.ln_2 = GPT2LayerNorm(d_model)
        self.mlp  = GPT2MLP(d_model)

    def call(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x