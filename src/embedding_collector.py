import sys
from transformers import T5Tokenizer, T5EncoderModel, GPT2Tokenizer, GPTNeoModel,XLNetModel, XLNetTokenizer, AlbertTokenizer, AlbertModel, RobertaTokenizer, RobertaModel
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


#EX_ID should be -2 for ROBERTA, ALBERT, T5; -1 for Neo; -3 for XLNet
#BOS added to Neo and XLNet (autoregressives)
if sys.argv[2] == 't5':
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5EncoderModel.from_pretrained("t5-base")
    EXTRACTION_ID = -2
    ADD_BOS = False

elif sys.argv[2] == 'albert':
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    model = AlbertModel.from_pretrained("albert-base-v2")
    EXTRACTION_ID = -2
    ADD_BOS = False

elif sys.argv[2] == 'gptneo':
    model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    EXTRACTION_ID = -1
    ADD_BOS = True

elif sys.argv[2] == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    EXTRACTION_ID = -2
    ADD_BOS = False

elif sys.argv[2] == 'xlnet':
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    model = XLNetModel.from_pretrained("xlnet-base-cased")
    EXTRACTION_ID = -3
    ADD_BOS = True

WRITE_MODEL = sys.argv[2]
model.eval()
print(tokenizer.bos_token)


if sys.argv[1] == "experiment2":
    #Get experiment 2 target embeddings
    with open(f'../data/experiment-phrases/experiment2.txt','r') as reader:
        k = reader.read().split('\n')

    tups = [i.split(',') for i in k]
    targets = [i[0] for i in tups]

    targets = [i for i in targets if 'feminine' not in i and 'masculine' not in i and 'jewish' not in i]
    print(len(targets))

    embs = []
    write_ = []
    print("obtaining embeddings for experiment 2")
    for idx,word in enumerate(tqdm(targets)):
        if word == '' or word == ' ' or word == '\n':
            continue
        with torch.no_grad():
            if ADD_BOS:
                input = tokenizer(f'{tokenizer.bos_token}{word}',return_tensors='pt')
            else:
                input = tokenizer(word,return_tensors='pt')
            embedding = model(**input)[0].numpy().squeeze(0)[EXTRACTION_ID]
        embs.append(embedding)
        write_.append(word)

    emb_arr = np.array(embs)
    emb_df = pd.DataFrame(emb_arr,index=write_)
    emb_df.to_csv(f'../output/embs_{WRITE_MODEL}.vec',sep=' ')

elif sys.argv[1] == "experiment3":
    #Get permutation embeddings (experiment 3)
    SOURCE_ = f'../data/experiment-phrases/experiment3.txt'

    with open(SOURCE_,'r') as reader:
        perm_ = reader.read().split('\n')
    all_contexts = [i.split(',')[0] for i in perm_]

    embs,write_ = [],[]
    print("obtaining embeddings for experiment 3")
    for idx,word in enumerate(tqdm(all_contexts)):
        if word == '' or word == ' ' or word == '\n':
            continue
        # if idx % 100 == 0 and idx > 0:
        #     print(idx)
        #     print(embedding.shape)
        with torch.no_grad():
            if ADD_BOS:
                input = tokenizer(f'{tokenizer.bos_token}{word}',return_tensors='pt')
            else:
                input = tokenizer(word,return_tensors='pt')
            embedding = model(**input)[0].numpy().squeeze(0)[EXTRACTION_ID]

        embs.append(embedding)
        write_.append(word)

    emb_arr = np.array(embs)
    emb_df = pd.DataFrame(emb_arr,index=write_)
    emb_df.to_csv(f'../output/permutation_embs_{WRITE_MODEL}.vec',sep=' ')