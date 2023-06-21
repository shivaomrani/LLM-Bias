from transformers import RobertaTokenizer, RobertaModel, GPT2Tokenizer, GPTNeoModel, XLNetModel, XLNetTokenizer, AlbertTokenizer, AlbertModel, T5Tokenizer, T5EncoderModel
import numpy as np
import pandas as pd
import sys

if sys.argv[2] == 't5':
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5EncoderModel.from_pretrained("t5-base")

elif sys.argv[2] == 'albert':
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    model = AlbertModel.from_pretrained("albert-base-v2")

elif sys.argv[2] == 'gptneo':
    model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')

elif sys.argv[2] == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")

elif sys.argv[2] == 'xlnet':
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    model = XLNetModel.from_pretrained("xlnet-base-cased")

model.eval()

WRITE_MODEL = sys.argv[2]

age = ['young','old']
height = ['tall','short']
education = ['educated','ignorant']
literacy = ['literate','illiterate']
race = ['white','black']
religion = ['christian','muslim']
sex = ['male','female']
weight = ['thin','fat']
intelligence = ['smart','stupid']
affluence = ['affluent','destitute']
orientation = ['heterosexual','homosexual']
gender = ['cisgender','transgender']
all = [age,height,education,literacy,race,religion,sex,weight,intelligence,affluence,orientation,gender]
# all = [age,weight,height,intelligence,education,literacy,affluence,race,orientation,religion,gender,sex]

group_dict, normed_dict = {},{}

#Print LaTeX to generate differential valence means graph
projection_df = pd.read_csv(f'../output/projection_products_{WRITE_MODEL}.csv',index_col=0)
phrases = projection_df.index.tolist()

for i,tup in enumerate(all):
    A = projection_df.loc[[i for i in phrases if tup[0] in i]].to_numpy()
    B = projection_df.loc[[i for i in phrases if tup[1] in i]].to_numpy()

    a_mean = np.mean(A)
    b_mean = np.mean(B)

    if a_mean >= b_mean:
        write_a = f'\\node[dot2,label=below:{tup[1]}] at (1.5+{i}*\\varc,{b_mean}*\\vara) ({tup[1]}){{}};'
        write_b = f'\\node[dot,label=above:{tup[0]}] at (1.5+{i}*\\varc,{a_mean}*\\vara) ({tup[0]}){{}};'
        write_c = f'\draw[-, green] ({tup[1]}.south) -- ({tup[0]}.north);'

    else:
        write_a = f'\\node[dot2,label=below:{tup[0]}] at (1.5+{i}*\\varc,{a_mean}*\\vara) ({tup[0]}){{}};'
        write_b = f'\\node[dot,label=above:{tup[1]}] at (1.5+{i}*\\varc,{b_mean}*\\vara) ({tup[1]}){{}};'
        write_c = f'\draw[-, green] ({tup[0]}.south) -- ({tup[1]}.north);'

    print(write_a)
    print(write_b)
    print(write_c)
