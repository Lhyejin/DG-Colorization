# -*- coding: utf-8 -*-
import random
from random import shuffle
import cv2
import torch
import numpy as np
import yaml
from sklearn.cluster import KMeans
import os, sys

def load_config(config_file):
    # Default parameters
    config_dict = {}
    config_dict['epochs'] = 1
    config_dict['lr'] = 0.0001
    config_dict['save_dir'] = './model/'
    config_dict['data_dir'] = './pacs_data'
    config_dict['name_dir'] = './pacs_data'
    config_dict['source_domain'] = ['photo']
    config_dict['save_freq'] = 10
    config_dict['num_cluster'] = 7
    with open(config_file, 'r') as conf:
        y = yaml.load(conf)

        if 'epochs' in y:
            config_dict['epochs'] = y['epochs']
        if 'lr' in y:
            config_dict['lr'] = y['lr']
        if 'batch_size' in y:
            config_dict['batch_size'] = y['batch_size']
        if 'save_dir' in y:
            config_dict['save_dir'] = y['save_dir']
        if 'data_dir' in y:
            config_dict['data_dir'] = y['data_dir']
        if 'name_dir' in y:
            config_dict['name_dir'] = y['name_dir']
        if 'source_domain' in y:
            config_dict['source_domain'] = y['source_domain']
        if 'save_freq' in y:
            config_dict['save_freq'] = y['save_freq']
        if 'train_dir' in y:
            config_dict['train_dir'] = y['train_dir']
        if 'val_dir' in y:
            config_dict['val_dir'] = y['val_dir']
        if 'weight_adv_s' in y:
            config_dict['weight_adv_s'] = y['weight_adv_s']
        if 'num_cluster' in y:
            config_dict['num_cluster'] = y['num_cluster']

    return config_dict




def split_data(data_dir, include_test=False, store_dir='./'):
    os.makedirs(store_dir, exist_ok=True)
    print(len(os.listdir(data_dir)))
    names = [f for f in os.listdir(data_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    num_samples = len(names)
    print('num_samples: ', len(names))

    if include_test:
        num_train_samples = int(num_samples * 0.88)
        num_test_samples = int(num_samples * 0.1)
    else:
        num_train_samples = int(num_samples * 0.992)
        num_test_samples = 0

    num_valid_samples = num_samples - num_train_samples - num_test_samples
    print('num_valid_samples: ' + str(num_valid_samples))
    valid_names = random.sample(names, num_valid_samples)
    test_names = []
    names_wo_valid = [n for n in names if n not in valid_names]
    if include_test:
        test_names = random.sample(names_wo_valid, num_test_samples)
        shuffle(test_names)
        with open(os.path.join(store_dir,  'test_names.txt'), 'w') as file:
            file.write('\n'.join(test_names))
    train_names = [n for n in names_wo_valid if n not in test_names]
    shuffle(valid_names)
    shuffle(train_names)
    with open(os.path.join(store_dir, 'valid_names.txt'), 'w') as file:
        file.write('\n'.join(valid_names))
    with open(os.path.join(store_dir, 'train_names.txt'), 'w') as file:
        file.write('\n'.join(train_names))

    print('num_train_samples: ', len(train_names))
    print('num_test_samples: ', len(test_names))
    print('num_valid_samples: ', len(valid_names))


# train/val/test
def save_names(data_dir, store_dir, domain, mode='train'):
    os.makedirs(store_dir, exist_ok=True)
    class_names = [f for f in os.listdir(data_dir)]
    names = []
    for class_name in class_names:
        temp = [mode+ '/'+ domain + '/' + class_name + '/' + f for f in os.listdir(os.path.join(data_dir, class_name)) if
                f.lower().endswith('.jpg')]
        names += temp
    print('number of samples', len(names))

    with open(os.path.join(store_dir, domain + '_%s_names.txt'%mode), 'w') as file:
        file.write('\n'.join(names))


def split_each_mode(names, include_test=False):
    num_samples = len(names)
    num_train_samples = int(num_samples * 0.8)
    if include_test:
        num_test_samples = int(num_samples * 0.1)
    else:
        num_train_samples = int(num_samples * 0.9)
        num_test_samples = 0

    num_valid_samples = num_samples - num_train_samples - num_test_samples
    valid_names = random.sample(names, num_valid_samples)
    test_names = []
    names_wo_valid = [n for n in names if n not in valid_names]
    if include_test:
        test_names = random.sample(names_wo_valid, num_test_samples)
        shuffle(test_names)
        
    train_names = [n for n in names_wo_valid if n not in test_names]
    shuffle(valid_names)
    shuffle(train_names)

    return train_names, valid_names, test_names

def split_domain_data(data_dir, domain='photo', include_test=False, store_dir='./'):
    os.makedirs(store_dir, exist_ok=True)
    print(os.listdir(data_dir))
    class_names = os.listdir(data_dir)  # if f.lower().endswith('.jpg')]
    names = []
    train_names = []
    valid_names = []
    test_names = []
    for class_name in class_names:
        temp = [domain + '/' + class_name + '/' + f for f in os.listdir(os.path.join(data_dir, class_name)) if
                f.lower().endswith('.jpg')]
        temp_train_names, temp_valid_names, temp_test_names = split_each_mode(temp, include_test)
        train_names += temp_train_names
        valid_names += temp_valid_names
        test_names += temp_test_names
        names += temp
    
    print('num_samples: ', len(names))

    if include_test:
        with open(os.path.join(store_dir, domain + '_test_names.txt'), 'w') as file:
            file.write('\n'.join(test_names))
    with open(os.path.join(store_dir, domain + '_valid_names.txt'), 'w') as file:
        file.write('\n'.join(valid_names))
    with open(os.path.join(store_dir, domain + '_train_names.txt'), 'w') as file:
        file.write('\n'.join(train_names))
    
    print('num_train_samples: ', len(train_names))
    print('num_test_samples: ', len(test_names))
    print('num_valid_samples: ', len(valid_names))
    

if __name__ == '__main__':
    split_domain_data('../officehome/Art', domain='Art', include_test=False, store_dir='../officehome/names')
    split_domain_data('../officehome/Clipart', domain='Clipart', include_test=False, store_dir='../officehome/names')
    split_domain_data('../officehome/Real_World', domain='Real_World', include_test=False, store_dir='../officehome/names')