"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
# import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
from pdb import set_trace as stx

# ------------------------------------------
local_dir = "edu_fineweb1B"
# remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 10 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
# fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train[]")
fw = load_dataset("beomi/fineweb-edu-fortified-mini")
# stx()

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    lst = doc['train']['text']
    for sentence in tqdm(lst):
        tokens.extend(enc.encode_ordinary(sentence))
    # print(len(tokens), '-----------') # 9e8个
    tokens_np = np.array(tokens)
    # np.save('tokens.npy', tokens_np)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
# nprocs = max(1, os.cpu_count()//4)
# print(nprocs, '---------')
# with mp.Pool(nprocs) as pool:
shard_index = 0
# preallocate buffer to hold current shard
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None
    # for tokens in pool.imap(tokenize, fw, chunksize=16):
# tokens = tokenize(fw)
tokens = np.load('tokens.npy')
    # is there enough space in the current shard for the new tokens?
for l in range(0, len(tokens), shard_size):
    split = 'val' if l==0 else 'train'
    if l+shard_size > len(tokens):
        break
    write_datafile(os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{l//shard_size:02d}"), tokens[l:l+shard_size])
