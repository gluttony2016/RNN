import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences


def build_dict(content_list, base_dir):
    print('----build character dictionary')
    char_dict = dict()
    char_dict['<PAD>'] = len(char_dict)
    char_dict['<START>'] = len(char_dict)
    char_dict['<EOS>'] = len(char_dict)
    char_dict['<UNK>'] = len(char_dict)

    char_num = dict()
    for content in content_list:
        for c in content.split():
            char_num[c] = char_num.get(c, 0) + 1

    # json.dump(char_num, open(base_dir+'data/gigaword/char_num.json', 'w'), indent=4, ensure_ascii=False)
    
    for content in content_list:
        for c in content.split():
            if c not in char_dict:
                if char_num[c] > 84:
                    char_dict[c] = len(char_dict)

    json.dump(char_dict, open(base_dir+'data/gigaword/char_dict.json', 'w'), indent=4, ensure_ascii=False)
    return char_dict


def build_emd(char_dict, emd_dim, base_dir):
    print('----build embedding matrix')
    emd_weight = np.zeros(shape=(1, emd_dim)).astype(np.float32)
    emd_weight = np.concatenate((emd_weight, np.random.randn(len(char_dict)-1, emd_dim).astype(np.float32)), axis=0)
    np.save(base_dir+'data/gigaword/emd_weight.npy', emd_weight)
    embedding_filepath = 'data/gigaword/emd_weight.npy'
    return embedding_filepath


def preprocess(fname, mode, config):
    print('Process file {} in mode {}'.format(fname, mode))
    f = open(fname, 'r')
    data = json.load(f)
    summary_list = list()
    text_list = list()
    for item in data:
        summary_list.append(item['title'].strip())
        text_list.append(item['content'].strip())

    if mode == 'train':
        char_dict = build_dict(text_list, config.base_dir)
        config.update_config('word_num', len(char_dict))

        embedding_filepath = build_emd(char_dict, config.config['emd_dim'], config.base_dir)
        config.update_config('embedding_filepath', embedding_filepath)
        config.save_config()
    else:
        char_dict = json.load(open(config.base_dir+'data/gigaword/char_dict.json', 'r'))

    print('----converting data')
    source = list()
    source_length = list()
    target = list()
    target_length = list()
    for text in text_list:
        feature = [char_dict.get(c, char_dict['<UNK>']) for c in text.split()]
        feature.append(char_dict['<EOS>'])

        source_length.append(min(len(feature), config.config['source_max_seq_length']))
        source.append(feature)

    source = pad_sequences(source,
            maxlen=config.config['source_max_seq_length'],
            dtype='int32',
            padding='post',
            truncating='post',
            value=char_dict['<PAD>'])

    for summary in summary_list:
        feature = [char_dict.get(c, char_dict['<UNK>']) for c in summary.split()]
        feature.append(char_dict['<EOS>'])

        target_length.append(min(len(feature), config.config['target_max_seq_length']))
        target.append(feature)

    target = pad_sequences(target,
            maxlen=config.config['target_max_seq_length'],
            dtype='int32',
            padding='post',
            truncating='post',
            value=char_dict['<PAD>'])

    np.save(config.base_dir+'data/gigaword/'+mode+'_source.npy', source)
    np.save(config.base_dir+'data/gigaword/'+mode+'_target.npy', target)
    np.save(config.base_dir+'data/gigaword/'+mode+'_source_length.npy', source_length)
    np.save(config.base_dir+'data/gigaword/'+mode+'_target_length.npy', target_length)
