import tensorflow as tf
import numpy as np
from model import DRGD
from tqdm import tqdm
from train_op import build_train_op
from loss import cross_entropy_sequence_loss
import os
from metrics import compute_metric_score
import json
import tensorflow.contrib.seq2seq as seq2seq

tf.logging.set_verbosity(tf.logging.INFO)

def train(source_helper, target_input_helper, target_output_helper, valid_source_helper, valid_target_output_helper, char_dict):
    model = DRGD() 
    model.build()

    emd_weight = np.load(model.base_dir+model.get_config('data', 'embedding_filepath'))
    embedding = tf.get_variable(initializer=emd_weight, name='embedding')

    source_sequence = tf.nn.embedding_lookup(embedding, model.source_placeholder)
    target_sequence = tf.nn.embedding_lookup(embedding, model.target_input_placeholder)

    encoder_output = model.encoder.encode(source_sequence, model.source_length)
    logits = model.decoder.decode(target_sequence, encoder_output)

    y_s = model.decoder.beam_search(encoder_output, embedding)

    target_mask = tf.sequence_mask(model.target_length, model.get_config('data', 'target_max_seq_length'), dtype=tf.float32)
    loss = seq2seq.sequence_loss(logits, model.target_output_placeholder, target_mask, average_across_timesteps=True, average_across_batch=True)

    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss)

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth=True

    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(model.get_config('train', 'epoch_num')):
            for step in range(source_helper.maximum_step):
                source_input, source_length = source_helper.next_batch()
                target_input, target_length = target_input_helper.next_batch()
                target_output, _ = target_output_helper.next_batch()

                _, l = sess.run([train_op, loss], feed_dict={
                    model.source_placeholder: source_input,
                    model.target_input_placeholder: target_input,
                    model.target_output_placeholder: target_output,
                    model.source_length: source_length,
                    model.target_length: target_length})

                tf.logging.log_every_n(level=tf.logging.INFO, msg='epoch {}/{}, step {}/{}, loss: {}'.format(epoch, model.get_config('train', 'epoch_num'), step, source_helper.maximum_step, l), n=10)
            final_output_all = list()
            target_output_all = list()
            for v_step in tqdm(range(valid_source_helper.maximum_step)):
                source_input, source_length = valid_source_helper.next_batch()
                target_output, _ = valid_target_output_helper.next_batch()

                final_output = sess.run(y_s, feed_dict={
                    model.source_placeholder: source_input,
                    model.source_length: source_length})
                        
                final_output_all.append(final_output)
                target_output_all.append(target_output)

                final_output = np.concatenate(final_output_all, axis=0)
                target_output = np.concatenate(target_output_all, axis=0)

            score = compute_metric_score(model.get_config('train', 'metric_path'), final_output, target_output, char_dict)
            tf.logging.info('Rough metric score :\n{}'.format(score))
            valid_source_helper.reset_cursor()
            valid_target_output_helper.reset_cursor()
