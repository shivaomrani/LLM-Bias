from transformers import GPT2Tokenizer, GPTNeoModel, XLNetModel, XLNetTokenizer, AlbertTokenizer, AlbertModel, T5Tokenizer, T5EncoderModel, RobertaTokenizer, RobertaModel
import numpy as np
import pandas as pd
import torch
import sys
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from weat_functions import SC_WEAT, my_svc


#EX_ID should be -2 for ROBERTA, ALBERT, T5; -1 for Neo; -3 for XLNet
#BOS added to Neo and XLNet (autoregressives)

if sys.argv[2] == 't5':
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5EncoderModel.from_pretrained("t5-base")
    EXTRACTION_ID = -2
    ADD_BOS = False
    LAYERS = 13

elif sys.argv[2] == 'albert':
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    model = AlbertModel.from_pretrained("albert-base-v2")
    EXTRACTION_ID = -2
    ADD_BOS = False
    LAYERS = 13

elif sys.argv[2] == 'gptneo':
    model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    EXTRACTION_ID = -1
    ADD_BOS = True
    LAYERS = 25

elif sys.argv[2] == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    EXTRACTION_ID = -2
    ADD_BOS = False
    LAYERS = 13

elif sys.argv[2] == 'xlnet':
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    model = XLNetModel.from_pretrained("xlnet-base-cased")
    EXTRACTION_ID = -3
    ADD_BOS = True
    LAYERS = 13

model.eval()
print(tokenizer.bos_token)


WRITE_MODEL = sys.argv[2]

#Define WEAT categories
pleasant = ['caress','freedom','health','love','peace','cheer','friend','heaven','loyal','pleasure','diamond','gentle','honest','lucky','rainbow','diploma','gift','honor','miracle','sunrise','family','happy','laughter','paradise','vacation']
unpleasant = ['abuse','crash','filth','murder','sickness','accident','death','grief','poison','stink','assault','disaster','hatred','pollute','tragedy','divorce','jail','poverty','ugly','cancer','kill','rotten','vomit','agony','prison']

group_dict, normed_dict = {},{}

#Read in Bellezza lexicon
k = pd.read_csv(f'../data/valnorm/Bellezza_Lexicon.csv',index_col='word')
bellezza_words = k.index.tolist()
bellezza_pleasantness = k['combined_pleasantness'].tolist()

#Define lists to hold embeddings
pleasant_embs,unpleasant_embs,target_embs = [],[],[]

#Create valence labels for learning a direction
valence_pairs = []
for i in range(len(pleasant)):
    valence_pairs.append([pleasant[i], unpleasant[i]])

#Get pleasant/unpleasant embeddings
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

#Learn a valence direction
clf_targets = np.concatenate((np.array(pleasant_embs),np.array(unpleasant_embs)))
labels = [1 for _ in range(25)] + [0 for _ in range(25)]

valence_vector, clf = my_svc(valence_pairs,(np.concatenate((pleasant_embs,unpleasant_embs),axis=0)))

#Get Bellezza embeddings
for word in bellezza_words:
    with torch.no_grad():
        if ADD_BOS:
            input = tokenizer(f'{tokenizer.bos_token}{word}',return_tensors='pt')
        else:
            input = tokenizer(word,return_tensors='pt')
        embedding = model(**input)[0].numpy().squeeze(0)[EXTRACTION_ID]
    target_embs.append(embedding)

#Projection ValNorm
associations = [valence_vector @ emb for emb in target_embs]
print(pearsonr(associations,bellezza_pleasantness))

#Layerwise ValNorm
dot,trad = [],[]
pleasant_embs,unpleasant_embs,target_embs = [],[],[]

for word in pleasant:
    with torch.no_grad():
        if ADD_BOS:
            input = tokenizer(f'{tokenizer.bos_token}{word}',return_tensors='pt')
        else:
            input = tokenizer(word,return_tensors='pt')
        embedding = model(**input,output_hidden_states=True)[-1]
    pleasant_embs.append(embedding)

for word in unpleasant:
    with torch.no_grad():
        if ADD_BOS:
            input = tokenizer(f'{tokenizer.bos_token}{word}',return_tensors='pt')
        else:
            input = tokenizer(word,return_tensors='pt')
        embedding = model(**input,output_hidden_states=True)[-1]
    unpleasant_embs.append(embedding)

for word in bellezza_words:
    with torch.no_grad():
        if ADD_BOS:
            input = tokenizer(f'{tokenizer.bos_token}{word}',return_tensors='pt')
        else:
            input = tokenizer(word,return_tensors='pt')
        embedding = model(**input,output_hidden_states=True)[-1]
    target_embs.append(embedding)

for layer in range(LAYERS):

    pleasant_,unpleasant_,target_ = [],[],[]

    pleasant_ = np.array([emb[layer].numpy().squeeze(0)[EXTRACTION_ID] for emb in pleasant_embs])
    unpleasant_ = np.array([emb[layer].numpy().squeeze(0)[EXTRACTION_ID] for emb in unpleasant_embs])
    target_ = np.array([emb[layer].numpy().squeeze(0)[EXTRACTION_ID] for emb in target_embs])

    clf_targets = np.concatenate((np.array(pleasant_),np.array(unpleasant_)))
    labels = [1 for _ in range(25)] + [0 for _ in range(25)]

    valence_vector, clf = my_svc(valence_pairs,(np.concatenate((pleasant_,unpleasant_),axis=0)))

    associations = [t @ valence_vector for t in target_]
    dot_correlation = pearsonr(associations,bellezza_pleasantness)[0]
    dot.append(dot_correlation)

    associations = [SC_WEAT(t,np.array(pleasant_),np.array(unpleasant_)) for t in target_]
    trad_correlation = pearsonr(associations,bellezza_pleasantness)[0]
    trad.append(trad_correlation)

dotprint = ' '.join([f'({i}, {dot[i]})' for i in range(len(dot))])
tradprint = ' '.join([f'({i}, {trad[i]})' for i in range(len(trad))])

print(dotprint)
print(tradprint)

layers = [i for i in range(LAYERS)]
plt.plot(layers,dot,label='SVC ValNorm',marker='o')
plt.plot(layers,trad,label='Traditional ValNorm',marker='o')
plt.legend()
plt.show()