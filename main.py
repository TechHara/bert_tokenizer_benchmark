import argparse
import os
from transformers import BertTokenizerFast, BertTokenizer
from datasets import load_dataset
import blingfire
import tensorflow_text as tf_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['hf', 'tf', 'bf'])
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['wikitext'])
    parser.add_argument('-n', type=int, default=1000000)

    args = parser.parse_args()
    if not ((args.input is None) ^ (args.dataset is None)):
        print('one of --input or --dataset must be provided')
        exit(-1)

    if args.method == 'hf':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenize = lambda s: tokenizer([s]).input_ids[0][1:-1]
    elif args.method == 'tf':
        vocab = []
        with open('vocab.txt') as f:
            vocab = f.read().split()
        tokenizer = tf_text.FastBertTokenizer(vocab, lower_case_nfd_strip_accents=True, )
        tokenize = lambda s: tokenizer.tokenize([s]).numpy()[0]
    elif args.method == 'bf':
        h = blingfire.load_model(os.path.join(os.path.dirname(blingfire.__file__), "bert_base_tok.bin"))
        tokenize = lambda s: blingfire.text_to_ids(h, s, 1575, 100, True)
    
    if args.input is not None:
        with open(args.input, 'r') as f:
            for line in f:
                print(' '.join(str(idx) for idx in tokenize(line)))
    
    else: #args.dataset is not None
        if args.dataset == 'wikitext':
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')['train']
        for n,text in enumerate(dataset):
            line = text['text']
            print(' '.join(str(idx) for idx in tokenize(line)))
            if args.n is not None and n >= args.n:
                break

    

if __name__ == '__main__':
    main()