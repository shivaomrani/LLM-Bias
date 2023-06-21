import pandas as pd
from weat_functions import SC_WEAT_Projection

#Obtain WEAT results by transformer and write to a LaTeX table
transformers_ = ['albert','gptneo','roberta','t5','xlnet']
write_string = ''


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

#Experiment 2 - Valence Biases
for tup in all:

    write_string += f'{tup[0]} vs. {tup[1]} & '

    for t_ in transformers_:

        projection_df = pd.read_csv(f'../output/projection_products_{t_}.csv',index_col=0)
        phrases = projection_df.index.tolist()[70:]
        phrases = [i for i in phrases if type(i) == str and 'jewish' not in i]

        A = projection_df.loc[[i for i in phrases if tup[0] in i]].to_numpy()
        B = projection_df.loc[[i for i in phrases if tup[1] in i]].to_numpy()
        es, p = SC_WEAT_Projection(A,B,1000)

        es = round(es,2)
        es_grey = 0
        if es > 0:
            es_grey = min(100,es*100)

        p_write = str(p)
        if p >= .05:
            p_write = '$n.s.$'
        if p < .05:
            p_write = '$.05$'
        if p < .01:
            p_write = '$.01$'
        for i in range(3,31):
            if p < 10 ** -i:
                p_write = '$10^{' + f'{-i}' + '}$'

        if t_ == 'xlnet':
            write_string += f'\cellcolor{{gray!{es_grey}}}${es}$  & {p_write}'
        else:
            write_string += f'\cellcolor{{gray!{es_grey}}}${es}$  & {p_write} & '

    write_string += r' \\ '
    write_string += f'\n'

print(write_string)