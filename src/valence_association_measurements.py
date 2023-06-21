from transformers import AlbertTokenizer, AlbertModel, GPT2Tokenizer, GPTNeoModel, XLNetModel, XLNetTokenizer, T5Tokenizer, T5EncoderModel, RobertaTokenizer, RobertaModel
import numpy as np
import pandas as pd
import torch
import sys
from weat_functions import my_svc


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

model.eval()

WRITE_MODEL = sys.argv[2]

pleasant = ['caress','freedom','health','love','peace','cheer','friend','heaven','loyal','pleasure','diamond','gentle','honest','lucky','rainbow','diploma','gift','honor','miracle','sunrise','family','happy','laughter','paradise','vacation']
unpleasant = ['abuse','crash','filth','murder','sickness','accident','death','grief','poison','stink','assault','disaster','hatred','pollute','tragedy','divorce','jail','poverty','ugly','cancer','kill','rotten','vomit','agony','prison']


#Learn the valence vector
group_dict, normed_dict = {},{}
pleasant_embs,unpleasant_embs = [],[]

valence_pairs = []
for i in range(len(pleasant)):
    valence_pairs.append([pleasant[i], unpleasant[i]])

print(tokenizer.bos_token)

for word in pleasant:
    with torch.no_grad():
        if ADD_BOS:
            input = tokenizer(f'{tokenizer.bos_token}{word}',return_tensors='pt')
        else:
            input = tokenizer(word,return_tensors='pt')
        embedding = model(**input)[0].numpy().squeeze(0)[EXTRACTION_ID]
    pleasant_embs.append(embedding)

for word in unpleasant:
    with torch.no_grad():
        if ADD_BOS:
            input = tokenizer(f'{tokenizer.bos_token}{word}',return_tensors='pt')
        else:
            input = tokenizer(word,return_tensors='pt')
        embedding = model(**input)[0].numpy().squeeze(0)[EXTRACTION_ID]
    unpleasant_embs.append(embedding)

clf_targets = np.concatenate((np.array(pleasant_embs),np.array(unpleasant_embs)))
labels = [1 for _ in range(25)] + [0 for _ in range(25)]

valence_pairs = []
for i in range(len(pleasant)):
    valence_pairs.append([pleasant[i], unpleasant[i]])

valence_vector, clf = my_svc(valence_pairs,(np.concatenate((pleasant_embs,unpleasant_embs),axis=0)))
np.save(f'../output/valence_vector_{WRITE_MODEL}.npy',valence_vector)


if sys.argv[1] == "experiment2":
    #expriment 2: Get fixed order embedding valence projections
    embs = pd.read_csv(f'../output/embs_{WRITE_MODEL}.vec',index_col=0,sep=' ')
    arr = embs.to_numpy()

    projection_products = valence_vector @ arr.T
    projection_df = pd.DataFrame(projection_products,index=embs.index.tolist(),columns=['projection'])
    projection_df.to_csv(f'../output/projection_products_{WRITE_MODEL}.csv')


elif sys.argv[1] == "experiment3":
    #experiment 3: Get permutation embedding valence projections
    embs = pd.read_csv(f'../output/permutation_embs_{WRITE_MODEL}.vec',index_col=0,sep=' ')
    arr = embs.to_numpy()

    projection_products = valence_vector @ arr.T
    projection_df = pd.DataFrame(projection_products,index=embs.index.tolist(),columns=['projection'])
    projection_df.to_csv(f'../output/projection_products_permutations_{WRITE_MODEL}.csv')