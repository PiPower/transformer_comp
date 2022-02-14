import numpy as np

from transformer_layers import *

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff, "gelu")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1 = self.mha1(x, x, x, mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out2 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training, mask)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x

class GPT2(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, sep_token ,rate=0.1):
        super().__init__()

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.sep_token = sep_token
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def train_call(self, input_tensor, training):
        mask = self.create_masks(input_tensor)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(input_tensor, training, mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output

    def infecrence_call(self, inputs):
        input_tensor, tar = inputs
        shape = input_tensor.shape

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(input_tensor, tar)

        enc_output = self.encoder(input_tensor, False, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(tar, enc_output, False, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        final_output = tf.argmax(final_output, axis = 2 )
        final_output = tf.slice(final_output, [0, final_output.shape[1]-1], [shape[0] , 1])

        return final_output

    def call(self, inputs, training):
        if training == True:
            return self.train_call(inputs, training)
        else:
            return self.infecrence_call(inputs)
    def gpt2_ahead_mask(self, inp):


        y = tf.ones(shape= tf.shape(inp), dtype=tf.int32) * self.sep_token
        z = tf.cast(tf.math.equal(inp,y), dtype=tf.int32)
        z = np.asarray( tf.argmax(z, axis=-1) ) + 1
        z  = z[ 0 ]
        mask = np.ones((183,183))
        mask = mask[: ]
        return z

    def create_masks(self,inp):
        mask = create_look_ahead_mask(tf.shape(inp)[1])
        return mask

    def train_step(self, input_list):
        inp_tensor, tar = input_list
        inp = inp_tensor[:, 1:]
        tar_real = inp_tensor[:, 1:]

        mask = tf.ones( tf.shape(inp), dtype=tf.int32) * self.sep_token
        mask_indicator = np.asarray(tf.equal(inp, mask))
        with tf.GradientTape() as tape:
            predictions = self(inp,training=True)
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')

            mask = tf.math.logical_not(tf.math.equal(tar_real, 0))
            loss_ = loss_object(tar_real, predictions)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            loss = self.compiled_loss(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(tar_real, predictions)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


