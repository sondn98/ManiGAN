from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from src.miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
from pyvi import ViTokenizer
import torchvision.transforms.functional as F

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys, wrong_caps, \
     wrong_caps_len, wrong_cls_id = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]

    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    ## 
    w_sorted_cap_lens, w_sorted_cap_indices = \
        torch.sort(wrong_caps_len, 0, True)

    wrong_caps = wrong_caps[w_sorted_cap_indices].squeeze()
    wrong_cls_id = wrong_cls_id[w_sorted_cap_indices].numpy()

    if cfg.CUDA:
        wrong_caps = Variable(wrong_caps).cuda()
        w_sorted_cap_lens = Variable(w_sorted_cap_lens).cuda()
    else:
        wrong_caps = Variable(wrong_caps)
        w_sorted_cap_lens = Variable(w_sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys, wrong_caps, w_sorted_cap_lens, wrong_cls_id]


# def get_imgs(img_path, imsize, flip, x, y, bbox=None,
#              transform=None, normalize=None):
#     img = Image.open(img_path).convert('RGB')
#     width, height = img.size
#     if bbox is not None:
#         r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
#         center_x = int((2 * bbox[0] + bbox[2]) / 2)
#         center_y = int((2 * bbox[1] + bbox[3]) / 2)
#         y1 = np.maximum(0, center_y - r)
#         y2 = np.minimum(height, center_y + r)
#         x1 = np.maximum(0, center_x - r)
#         x2 = np.minimum(width, center_x + r)
#         img = img.crop([x1, y1, x2, y2])
#
#     if transform is not None:
#         img = transform(img)
#         ## crop
#         img = img.crop([x, y, x + 256, y + 256])
#         if flip:
#             img = F.hflip(img)
#
#     ret = []
#     if cfg.GAN.B_DCGAN:
#         ret = [normalize(img)]
#     else:
#         for i in range(cfg.TREE.BRANCH_NUM):
#             if i < (cfg.TREE.BRANCH_NUM - 1):
#                 re_img = transforms.Scale(imsize[i])(img)
#             else:
#                 re_img = img
#             ret.append(normalize(re_img))
#
#     return ret

def get_imgs(img_path, imsize, normalize=None):
    img = Image.open(img_path).convert('RGB')

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        # self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s' % (data_dir, filenames[i])
            cap_path = cap_path.replace('.png', '.txt')
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf-8').split('\n')
                # cnt = 0
                sentences = []
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokens = ViTokenizer.tokenize(cap.lower()).split(' ')

                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('utf-8', 'strict').decode('utf-8')
                        if len(t) > 0:
                            tokens_new.append(t)
                    sentences.append(tokens_new)
                    # cnt += 1
                    # if cnt == self.embeddings_num:
                    #     break
                # if cnt < self.embeddings_num:
                #     print('ERROR: the captions for %s less than %d'
                #           % (filenames[i], cnt))
                all_captions.append(sentences)
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for desc in captions:
            for sent in desc:
                for word in sent:
                    word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for train_desc in train_captions:
            train_desc_new = []
            for t in train_desc:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                # this train_captions_new hold index of each word in sentence
                train_desc_new.append(rev)
            train_captions_new.append(train_desc_new)

        test_captions_new = []
        for test_desc in test_captions:
            test_desc_new = []
            for t in test_desc:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                test_desc_new.append(rev)
            test_captions_new.append(test_desc_new)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'face_captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                print("filepath", filepath)
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f) #, encoding='latin1'
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, cap_idx):
        # a list of indices for a sentence
        sent_ix = random.randint(0, len(self.captions[cap_idx]))
        sent_caption = np.asarray(self.captions[cap_idx][sent_ix]).astype('int64')
        # if (sent_caption == 0).sum() > 0:
            # print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        # if self.bbox is not None:
        #     bbox = self.bbox[key]
        #     data_dir = self.data_dir
        # else:
        #     bbox = None
        #     data_dir = self.data_dir

        # flip = random.rand() > 0.5

        # new_w = new_h = int(256 * 76 / 64)
        # x = random.randint(0, np.maximum(0, new_w - 256))
        # y = random.randint(0, np.maximum(0, new_h - 256))

        img_name = '%s/VN_CELEB/images/%s' % (self.data_dir, key)
        imgs = get_imgs(img_name, self.imsize, normalize=self.norm)
        # random select a sentence
        # sent_ix = random.randint(0, self.embeddings_num)
        # new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(index)

        ###
        wrong_idx = random.randint(0, len(self.filenames))
        # wrong_new_sent_ix = wrong_idx * self.embeddings_num + sent_ix
        wrong_caps, wrong_cap_len = self.get_caption(wrong_idx)
        wrong_cls_id = self.class_id[wrong_idx]
        ###

        return imgs, caps, cap_len, cls_id, key, wrong_caps, \
            wrong_cap_len, wrong_cls_id

    def __len__(self):
        return len(self.filenames)
