import os
import time
import regex as re
import pandas as pd

if not os.path.exists('Data'):
   os.makedirs('Data')

start_time = time.time()
print(f"Preparing DiscoGeM corpus...")

discogem_columns = ['itemid',
                    'arg1',
                    'arg2',
                    'arg1_singlesentence',
                    'arg2_singlesentence',
                    'domconn_step1',
                    'domconn_step2',
                    'majority_softlabel',
                    'majority_distrlabel20',
                    'majority_distrlabel40',
                    'majoritylabel_sampled',
                    'reflabel']

df_discogem = pd.read_csv('Corpora/DiscoGeM/DiscoGeM_corpus/DiscoGeMcorpus_annotations_wide.csv', usecols = discogem_columns)
df_discogem = df_discogem[discogem_columns]

df_discogem.to_csv('Data/discogem.csv', index=False)
print(f'Completed in {(time.time()-start_time)/60:.2f} minutes.')

start_time = time.time()
print(f"Preparing DiscoGeM-Multi corpus...")

df_discogem_multi = df_discogem[['itemid', 'arg1', 'arg2', 'arg1_singlesentence', 'arg2_singlesentence','majoritylabel_sampled']].copy()

df_discogem_multi['arg1_arg2'] = df_discogem_multi['arg1'].copy() + ' ' + df_discogem_multi['arg2'].copy()
df_discogem_multi['sent1_sent2'] = df_discogem_multi['arg1_singlesentence'].copy() + ' ' + df_discogem_multi['arg2_singlesentence'].copy()
df_discogem_multi = df_discogem_multi[['itemid', 'arg1', 'arg2', 'arg1_singlesentence', 'arg2_singlesentence','arg1_arg2', 'sent1_sent2', 'majoritylabel_sampled']]

label_names = re.findall('[\w\-]+(?=\:)', df_discogem['majority_softlabel'].iloc[0])
for i in label_names:
    df_discogem_multi[i] = ''

for row in range(len(df_discogem['majority_softlabel'])):
    label_values = re.findall('(?<=:)\d\.?\d?', df_discogem['majority_softlabel'].iloc[row])
    for i in range(len(label_names)):
        df_discogem_multi[label_names[i]].iloc[row] = label_values[i]

df_discogem_multi.to_csv('Data/discogem_multi.csv', index=False)
print(f'Completed in {(time.time()-start_time)/60:.2f} minutes.')

start_time = time.time()
print(f"Preparing QADC corpus...")

files = os.listdir('Corpora/DiscoGeM/QADiscourse_annotations')
files_tsv = [f for f in files if f[-9:] == '_wide.tsv']
qadc_columns = ['itemid',
                'arg1',
                'arg2',
                'sent1',
                'sent2',
                'majority_softlabel',
                'majority_distrlabel20',
                'majority_distrlabel40',
                'majoritylabel_sampled',
                'majority_softlabel_qa',
                'majority_distrlabel20_qa',
                'majority_distrlabel40_qa',
                'majoritylabel_sampled_qa',
                'reflabel']

df_qadc = pd.concat([pd.read_csv('Corpora/DiscoGeM/QADiscourse_annotations/'+f, sep='\t') for f in files_tsv])
df_qadc = df_qadc[qadc_columns]

empty_rows = list(df_qadc['itemid'].loc[df_qadc['majority_softlabel'].isnull()])

for i in empty_rows:
    df_qadc['majority_softlabel'].loc[df_qadc['itemid'] == i] = df_qadc['majority_softlabel_qa'].loc[df_qadc['itemid'] == i]
    df_qadc['majority_distrlabel20'].loc[df_qadc['itemid'] == i] = df_qadc['majority_distrlabel20_qa'].loc[df_qadc['itemid'] == i]
    df_qadc['majority_distrlabel40'].loc[df_qadc['itemid'] == i] = df_qadc['majority_distrlabel40_qa'].loc[df_qadc['itemid'] == i]
    df_qadc['majoritylabel_sampled'].loc[df_qadc['itemid'] == i] = df_qadc['majoritylabel_sampled_qa'].loc[df_qadc['itemid'] == i]

df_qadc = df_qadc.drop(columns = ['majority_softlabel_qa', 'majority_distrlabel20_qa', 'majority_distrlabel40_qa', 'majoritylabel_sampled_qa'])

df_qadc.to_csv('Data/qadc.csv', index=False)
print(f'Completed in {(time.time()-start_time)/60:.2f} minutes.')

start_time = time.time()
print(f"Preparing QADC-Multi corpus...")

df_qadc_multi = df_qadc[['itemid', 'arg1', 'arg2', 'sent1', 'sent2', 'majoritylabel_sampled']].copy()

df_qadc_multi['arg1_arg2'] = df_qadc_multi['arg1'].copy() + ' ' + df_qadc_multi['arg2'].copy()
df_qadc_multi['sent1_sent2'] = df_qadc_multi['sent1'].copy() + ' ' + df_qadc_multi['sent2'].copy()
df_qadc_multi = df_qadc_multi[['itemid', 'arg1', 'arg2', 'sent1', 'sent2','arg1_arg2', 'sent1_sent2', 'majoritylabel_sampled']]

label_names = re.findall('[\w\-]+(?=\:)', df_qadc['majority_softlabel'].iloc[0])
for i in label_names:
    df_qadc_multi[i] = ''

for row in range(len(df_qadc['majority_softlabel'])):
    label_values = re.findall('(?<=:)\d\.?\d?', df_qadc['majority_softlabel'].iloc[row])
    for i in range(len(label_names)):
        df_qadc_multi[label_names[i]].iloc[row] = label_values[i]

df_qadc_multi.to_csv('Data/qadc_multi.csv', index=False)
print(f'Completed in {(time.time()-start_time)/60:.2f} minutes.')