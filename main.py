import argparse
# from data.LCSTS.preprocess import preprocess
from data.gigaword.preprocess import preprocess
import json
from config.configurable import Configurable
from train_op import build_train_op
from helper import Helper
from run import train

config = Configurable('data')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preprocess', help="preprocess train/dev/test data", action='store_true')
    parser.add_argument('-t', '--train', help="train DRGD model", action='store_true')
    # parser.add_argument('--test', help="test trained DRGD model", action='store_true')
    # parser.add_argument('--inference', help="inference based model", action='store_true')
    result = parser.parse_args()

    if result.preprocess:
        preprocess(fname=config.base_dir+config.config['train_file'], mode='train', config=config)
        preprocess(fname=config.base_dir+config.config['test_file'], mode='dev', config=config)

    if result.train:
        source_helper = Helper(data_filepath='./data/gigaword/train_source.npy', length_filepath='./data/gigaword/train_source_length.npy', mode='source')
        target_input_helper = Helper(data_filepath='./data/gigaword/train_target.npy', length_filepath='./data/gigaword/train_target_length.npy', mode='target_input')
        target_output_helper = Helper(data_filepath='./data/gigaword/train_target.npy', length_filepath='./data/gigaword/train_target_length.npy', mode='target_output')
        valid_source_helper = Helper(data_filepath='./data/gigaword/dev_source.npy', length_filepath='./data/gigaword/dev_source_length.npy', mode='source')
        valid_target_output_helper = Helper(data_filepath='./data/gigaword/dev_target.npy', length_filepath='./data/gigaword/dev_target_length.npy', mode='target_output')
        char_dict = json.load(open('./data/gigaword/char_dict.json', 'r'))
        train(source_helper, target_input_helper, target_output_helper, valid_source_helper, valid_target_output_helper, char_dict)
