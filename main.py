import argparse
from transformers import BertTokenizerFast, BertTokenizer
from datasets import load_dataset
import tensorflow_hub as hub
import tensorflow_text as tf_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['hfbt', 'hfbtf', 'tfbtf'])
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['wikitext'])

    args = parser.parse_args()
    if not ((args.input is None) ^ (args.dataset is None)):
        print('one of --input or --dataset must be provided')
        exit(-1)

    if args.method.startswith('hf'):
        Tokenizer = BertTokenizer if args.method == 'hfbt' else BertTokenizerFast
        tokenizer = Tokenizer.from_pretrained('bert-base-uncased')
        tokenize = lambda s: tokenizer([s]).input_ids[0][1:-1]
    elif args.method.startswith('tf'):
        vocab = []
        with open('vocab.txt') as f:
            for line in f:
                vocab.append(line.strip())
        tokenizer = tf_text.FastBertTokenizer(vocab, lower_case_nfd_strip_accents=True, )
        tokenize = lambda s: tokenizer.tokenize([s]).numpy()[0]

    
    if args.input is not None:
        with open(args.input, 'r') as f:
            for line in f:
                print(' '.join(str(idx) for idx in tokenize(line)))
    
    else: #args.dataset is not None
        if args.dataset == 'wikitext':
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')['train']
        for text in dataset:
            line = text['text']
            print(' '.join(str(idx) for idx in tokenize(line)))

    

if __name__ == '__main__':
    main()