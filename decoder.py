import tensorflow as tf
from config.configurable import Configurable
from collections import namedtuple
from pydoc import locate
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq


BeamSearchDecoderState = namedtuple("BeamSearchDecoderState", 
    ("cell_state", "log_probs", "finished")
)


class Decoder(Configurable):
    def __init__(self):
        super(Decoder, self).__init__('decoder')
        self.cell = self.build_cell(self.config['cell_classname'], 'cell') # GRU cell

    def decode_onestep(self, inputs, encoder_output, state):
        batch_size = self.get_config('train', 'batch_size')
        hidden_dim = self.config['cell']['num_units']
        word_num = self.get_config('data', 'word_num')
        emd_dim = self.get_config('data', 'emd_dim')
        source_max_seq_length = self.get_config('data', 'source_max_seq_length')

        source_mask = tf.sequence_mask(lengths=encoder_output.attention_values_length, maxlen=source_max_seq_length, dtype=tf.bool)

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
            W_d_hy = tf.get_variable(name='W_d_hy', shape=[hidden_dim, word_num], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_d_hy = tf.get_variable(name='b_d_hy', shape=[word_num], dtype=tf.float32, initializer=tf.zeros_initializer())

        a_ij = self.compute_attention_weight(state, encoder_output.attention_values, source_mask)
        c_t = tf.reduce_sum(tf.multiply(encoder_output.attention_values, a_ij), axis=1)

        _, state = self.cell(inputs=tf.concat([inputs, c_t], axis=-1), state=state)

        logit_t = tf.nn.xw_plus_b(state, W_d_hy, b_d_hy)

        return state, logit_t

    def decode(self, inputs, encoder_output):
        target_max_seq_length = self.get_config('data', 'target_max_seq_length')
        batch_size = self.get_config('train', 'batch_size')

        state = tf.reduce_mean(encoder_output.outputs, axis=1)
        logits = list()
        for step in range(target_max_seq_length):
            state, logit = self.decode_onestep(inputs[:, step, :], encoder_output, state)
            logits.append(logit)
        logits = tf.stack(logits, axis=1)

        return logits

    def beam_search(self, encoder_output, embedding):
        vocab_size = self.get_config('data', 'word_num')
        end_id = self.get_config('data', 'end_id')
        start_id = self.get_config('data', 'start_id')
        beam_width = self.config['beam_search_width']
        batch_size = self.get_config('train', 'batch_size')
        target_max_seq_length = self.get_config('data', 'target_max_seq_length')
        length_penalty_weight = self.config['length_penalty_weight']
        hidden_dim = self.config['cell']['num_units']

        start_tokens = tf.fill([batch_size, beam_width], start_id)
        start_inputs = tf.nn.embedding_lookup(embedding, start_tokens)
        inputs = start_inputs

        finished = tf.one_hot(
            indices=tf.zeros(batch_size, tf.int32),
            depth=beam_width,
            on_value=False,
            off_value=True,
            dtype=tf.bool) # batch_size, beam_width

        log_probs = tf.one_hot(
            indices=tf.zeros(batch_size, tf.int32),
            depth=beam_width,
            on_value=tf.convert_to_tensor(0.0, tf.float32),
            off_value=tf.convert_to_tensor(-np.Inf, tf.float32),
            dtype=tf.float32)

        init_state = tf.reduce_mean(encoder_output.outputs, axis=1)
        tile_state = tf.tile(tf.expand_dims(init_state, 1), [1, beam_width, 1])

        beam_state = BeamSearchDecoderState(
            cell_state=tile_state,
            log_probs=log_probs,
            # lengths=tf.zeros([batch_size, beam_width], dtype=tf.int32),
            finished=finished)

        y_s = tf.zeros([batch_size, beam_width, 0], dtype=tf.int32)

        for step in range(target_max_seq_length):
            state = beam_state.cell_state

            new_state = list()
            logits = list()
            for search_step in range(beam_width):
                state_t, logits_t = self.decode_onestep(inputs[:, search_step, :], encoder_output, state[:, search_step, :])
                new_state.append(state_t)
                logits.append(logits_t)

            states = tf.stack(new_state, 1) # batch_size, beam_width
            logits = tf.stack(logits, 1) # batch_size, beam_width, vocab_size

            # prediction_lengths = beam_state.lengths
            previously_finished = beam_state.finished

            step_mask = tf.to_float(tf.logical_not(previously_finished))
            step_log_probs = tf.nn.log_softmax(logits, axis=-1)
            # step_log_probs = self.mask_probs(step_log_probs, end_id, previously_finished)
            # step_log_scores, step_word_indices = tf.nn.top_k(step_log_probs, k=beam_width)

            step_log_scores = step_log_probs * tf.expand_dims(step_mask, 2)
            total_probs = tf.expand_dims(beam_state.log_probs, 2) + step_log_scores

            # lengths_to_add = tf.one_hot(
            #     indices=tf.fill([batch_size, beam_width], end_id),
            #     depth=vocab_size,
            #     on_value=tf.to_int32(0),
            #     off_value=tf.to_int32(1),
            #     dtype=tf.int32)
            # add_mask = tf.to_int32(tf.logical_not(previously_finished))
            # lengths_to_add = lengths_to_add * tf.expand_dims(add_mask, 2)
            # new_prediction_lengths = lengths_to_add + tf.expand_dims(prediction_lengths, 2)

            # scores = self.get_scores(log_probs=total_probs,
            #     sequence_lengths=new_prediction_lengths,
            #     length_penalty_weight=length_penalty_weight)

            scores_flat = tf.reshape(total_probs, (batch_size, -1))
            # word_indices_flat = tf.reshape(step_word_indices, (batch_size, -1))
            next_beam_scores, word_indices = tf.nn.top_k(scores_flat, k=beam_width, sorted=True)

            next_beam_probs = tf.batch_gather(tf.reshape(total_probs, (batch_size, -1)), word_indices) # batch_size, beam_width

            # next_word_ids = tf.batch_gather(word_indices_flat, tf.mod(word_indices, vocab_size))
            next_word_ids = tf.mod(word_indices, vocab_size)
            next_beam_ids = tf.to_int32(tf.div(word_indices, vocab_size))

            previously_finished = tf.batch_gather(previously_finished, next_beam_ids)
            next_finished = tf.logical_or(previously_finished,
                tf.equal(next_word_ids, end_id))
            # next_beam_probs = tf.batch_gather(total_probs, next_word_ids)

            # lengths_to_add = tf.to_int32(tf.logical_not(previously_finished))
            # next_prediction_len = tf.batch_gather(beam_state.lengths, next_beam_ids)
            # next_prediction_len = next_prediction_len + lengths_to_add

            next_states = tf.batch_gather(states, next_beam_ids)

            beam_state = BeamSearchDecoderState(
                cell_state=next_states,
                log_probs=next_beam_probs,
                # lengths=next_prediction_len,
                finished=next_finished)

            inputs = tf.cond(
                tf.reduce_all(next_finished),
                lambda: start_inputs, lambda: tf.nn.embedding_lookup(embedding, next_word_ids))

            y_s = tf.concat([tf.batch_gather(y_s, next_beam_ids), tf.expand_dims(next_word_ids, axis=2)], axis=2)
        return y_s[:, 0, :]


    def mask_probs(self, probs, eos_token, finished):
        vocab_size = probs.shape[-1]
        finished_row = tf.one_hot(
            indices=eos_token,
            depth=vocab_size,
            dtype=tf.float32,
            on_value=tf.convert_to_tensor(0., dtype=tf.float32),
            off_value=tf.convert_to_tensor(-np.Inf, dtype=tf.float32))

        finished_probs = tf.tile(
            tf.reshape(finished_row, [1, 1, -1]),
            tf.concat([finished.shape, [1]], 0)) # batch_size, beam_width, vocab_size

        finished_mask = tf.tile(
            tf.expand_dims(finished, 2), [1, 1, vocab_size]) # batch_size, beam_width, vocab_size

        return tf.where(finished_mask, finished_probs, probs)

    def get_scores(self, log_probs, sequence_lengths, length_penalty_weight):
        length_penalty_ = self.length_penalty(sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight)

        return log_probs / length_penalty_

    def length_penalty(self, sequence_lengths, penalty_factor):
        penalty_factor = tf.to_float(penalty_factor)
        return tf.div((tf.to_float(5.0) + tf.to_float(sequence_lengths))**penalty_factor, (tf.to_float(1.0 + 5.0))**penalty_factor)

    def build_cell(self, cell_classname, cell_name):
        cell_class = locate(cell_classname)
        return cell_class(num_units=self.config['cell']['num_units'],
                name=cell_name,
                **self.config['cell']['cell_params'])

    def compute_attention_weight(self, state, hidden_states, source_mask):
        encoder_hidden_hid = self.get_config('encoder', 'cell')['num_units']
        decoder_hidden_hid = self.config['cell']['num_units']

        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE) as scope:
            W_d_hh = tf.get_variable(name='W_d_hh', shape=[2*encoder_hidden_hid, decoder_hidden_hid], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            W_e_hh = tf.get_variable(name='W_e_hh', shape=[decoder_hidden_hid, decoder_hidden_hid], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b_a = tf.get_variable(name='b_a', shape=[decoder_hidden_hid], initializer=tf.zeros_initializer(), dtype=tf.float32)
            v = tf.get_variable(name='v', shape=[decoder_hidden_hid], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

        r_1 = tf.einsum('ijk,kl->ijl', hidden_states, W_e_hh)
        r_2 = tf.expand_dims(tf.matmul(state, W_d_hh), axis=1)
        a = tf.add(r_1, r_2)
        t = tf.tanh(tf.add(a, b_a))
        e_ij = tf.reduce_sum(tf.multiply(v, t), axis=-1)

        mask_value = tf.convert_to_tensor(-np.Inf, dtype=tf.float32)
        e_ij_mask = mask_value * tf.ones_like(e_ij)

        e_ij = tf.where(source_mask, e_ij, e_ij_mask) # replace the attention weight on <PAD> token with -inf, then after softmax it can be zero

        a_ij = tf.expand_dims(tf.nn.softmax(e_ij, axis=1), axis=-1)
        return a_ij
