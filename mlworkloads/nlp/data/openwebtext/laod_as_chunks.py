from tqdm.auto import tqdm
from datasets import load_dataset
import os
import pandas as pd
import random

ds = load_dataset("openwebtext")

# create necessary folders
os.mkdir('data')
os.mkdir('data/original')

#save text in chunks of 4000 samples
text = []
ind = 0

for sample in tqdm(ds['train']):
    # replace all newlines
    sample = sample['text'].replace('\n','')
    
    # append cleaned sample to all texts
    text.append(sample)
    
    # if we processed 4000 samples, write them to a file and start over
    if len(text) == 4000:
        with open(f"data/original/text_{ind}.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(text))
        text = []
        ind += 1

# write remaining samples to a file
with open(f"data/original/text_{ind}.txt", 'w', encoding='utf-8') as f:
    f.write('\\n'.join(text))