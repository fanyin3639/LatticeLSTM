# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-07-06 11:08:27

import argparse
import copy
import datetime
import gc
import os
import pickle as pickle
import random
import sys
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim

from model.bilstmcrf import BiLSTMCRF as SeqModel
from utils.data import Data
from utils.metric import get_ner_fmeasure

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data, gaz_file, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_gaz_file(gaz_file)
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    data.build_gaz_alphabet(test_file)
    data.fix_alphabet()
    return data


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "p:",pred, pred_tag.tolist()
        # print "g:", gold, gold_tag.tolist()
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)


def load_data_setting(save_dir):
    with open(os.path.join(save_dir, 'data.set'), 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_dir)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1

    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end >train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model(gaz_list,batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # print("tag_seq", tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results  


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    ## not reorder label
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(list(map(max, length_list)))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    
    ## keep the gaz_list in orignial order
    
    gaz_list = [ gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def train(data, save_model_dir, seg=True):
    print("Training model...")
    data.show_data_summary()
    model = SeqModel(data)
    print("finished built model.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)
    best_dev = -1
    data.HP_iteration = 100
    #  start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print(("Epoch: %s/%s" %(idx,data.HP_iteration)))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()
        batch_size = 1
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = min((batch_id+1)*batch_size, train_num)
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            gaz_list,  batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(gaz_list, batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data[0]
            total_loss += loss.data[0]
            batch_loss += loss

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token)))
                sys.stdout.flush()
                sample_loss = 0
            if end % data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token)))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (idx, epoch_cost, train_num/epoch_cost, total_loss)))
        speed, acc, p, r, f, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if seg:
            current_score = f
            print(("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f)))
        else:
            current_score = acc
            print(("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc)))

        if current_score > best_dev:
            if seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            model_name = save_model_dir + '.' + str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            best_dev = current_score 
        # ## decode test
        # speed, acc, p, r, f, _ = evaluate(data, model, "test")
        # test_finish = time.time()
        # test_cost = test_finish - dev_finish
        # if seg:
        #     print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f)))
        # else:
        #     print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc)))
        gc.collect() 


def load_model_decode(save_dir, data, name, seg=True):
    print("Load Model from file: ", save_dir)
    model = SeqModel(data)
    model.load_state_dict(torch.load(save_dir))
    print(F"Decode {name} data ...")
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f)))
    else:
        print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc)))
    return pred_results


if __name__ == '__main__':
    char_emb = '/data/nfsdata/nlp/embeddings/chinese/gigaword/gigaword_chn.all.a2b.uni.ite50.vec'
    bichar_emb = None
    # bichar_emb = '/data/nfsdata/nlp/embeddings/chinese/gigaword/gigaword_chn.all.a2b.bi.ite50.vec'
    # gaz_file = '/data/nfsdata/nlp/embeddings/chinese/ctb/ctb.50d.vec'  # NER
    gaz_file = '/data/nfsdata/nlp/embeddings/chinese/wiki/zh.wiki.bpe.vs200000.d50.w2v.txt'  # CWS
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CRF')
    parser.add_argument('--embedding',  help='Embedding for words', default='None')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--name', default='CTB9POS')
    parser.add_argument('--mode', default='char')
    parser.add_argument('--data_dir', default='/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/')
    parser.add_argument('--raw')
    parser.add_argument('--loadmodel')
    args = parser.parse_args()
    train_file = F'{args.data_dir}/{args.name}/train.{args.mode}.bmes'
    dev_file = F'{args.data_dir}/{args.name}/dev.{args.mode}.bmes'
    test_file = F'{args.data_dir}/{args.name}/test.{args.mode}.bmes'
    save_dir = F'/data/nfsdata/nlp/projects/{args.name}.{args.mode}.{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    print("Status:", args.status)
    print("Train file:", train_file)
    print("Dev file:", dev_file)
    print("Test file:", test_file)
    print("Raw file:", args.raw)
    print("Char emb:", char_emb)
    print("Bichar emb:", bichar_emb)
    print("Gaz file:",gaz_file)
    print("Save dir:", save_dir)
    sys.stdout.flush()
    
    if args.status == 'train':
        data = Data()
        data.HP_use_char = False
        data.HP_batch_size = 1
        data.use_bigram = True  # ner: False, cws: True
        data.gaz_dropout = 0.5
        data.HP_lr = 0.01  # cws
        data.HP_dropout = 0.5  # cws
        data.HP_iteration = 50  # cws
        data.norm_gaz_emb = True # ner: False, cws: True
        data.HP_fix_gaz_emb = False
        data_initialization(data, gaz_file, train_file, dev_file, test_file)
        data.generate_instance_with_gaz(train_file, 'train')
        data.generate_instance_with_gaz(dev_file, 'dev')
        data.generate_instance_with_gaz(test_file, 'test')
        data.build_word_pretrain_emb(char_emb)
        data.build_biword_pretrain_emb(bichar_emb)
        data.build_gaz_pretrain_emb(gaz_file)
        torch.save(data, save_dir + '/data.set')
        data = torch.load(save_dir + '/data.set')
        train(data, save_dir)
    elif args.status == 'test':
        data = load_data_setting(args.loadmodel + '/data.set')
        data.generate_instance_with_gaz(dev_file, 'dev')
        load_model_decode(args.loadmodel + '/saved.model', data, 'dev')
        data.generate_instance_with_gaz(test_file, 'test')
        load_model_decode(args.loadmodel + '/saved.model', data, 'test')
    elif args.status == 'decode':
        data = load_data_setting(args.loadmodel + '/data.set')
        data.generate_instance_with_gaz(args.raw, 'raw')
        decode_results = load_model_decode(args.loadmodel + '/saved.model', data, 'raw')
        data.write_decoded_results(args.loadmodel + '/decoded.output', decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
