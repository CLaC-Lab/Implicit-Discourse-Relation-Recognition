import os
import csv
import time
import wandb
import pandas as pd
import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification

MODEL_NAME = 'roberta-base'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformers.logging.set_verbosity_error() # remove model checkpoint warning

EPOCHS = 10
BATCH_SIZE = 64
THREADS = 0

LR = 1e-5
DECAY = 0.1

# define matplotlib font settings
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.family': 'Times New Roman'})

class IDRR_Dataset(Dataset):
    
    def __init__(self, x, y): 
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        item_x_input_ids = self.x['input_ids'][index]
        item_x_attention_mask = self.x['attention_mask'][index]
        item_y = torch.FloatTensor(self.y.iloc[index])
        return {'input_ids': item_x_input_ids, 'attention_mask':item_x_attention_mask, 'labels': item_y}
    
    def __len__(self):
        return len(self.x['input_ids'])

# initialize wandb tracker
wandb.login()
run = wandb.init(project = MODEL_NAME, config = {'Epochs': EPOCHS, 'Batch Size': BATCH_SIZE, 'Learning Rate': LR, 'Decay': DECAY})

# read data
df = pd.read_csv('Data/discogem_multi.csv')

'''
# plot histogram for the character length of ARG1+ARG2
arg1_arg2_string = '\n'.join((f'Ave: %d' % df['arg1_arg2'].str.len().mean(), f'Max: %d' % df['arg1_arg2'].str.len().max(), f'Min: %d' % df['arg1_arg2'].str.len().min()))
sent1_sent2_string = '\n'.join((f'Ave: %d' % df['sent1_sent2'].str.len().mean(), f'Max: %d' % df['sent1_sent2'].str.len().max(), f'Min: %d' % df['sent1_sent2'].str.len().min()))
figure, (axis_1, axis_2) = plt.subplots(2, 1, figsize=(3, 3))
axis_1.set_title('ARG1+ARG2', fontweight='bold')
axis_1.set_xlabel('Character length')
axis_1.set_xlim((0, 800))
axis_1.set_xticks([0, 200, 400, 600, 800])
axis_1.set_ylim((0, 400))
axis_1.hist(df['sent1_sent2'].str.len(), bins=50, color='tab:green')
axis_1.axvline(round(df['sent1_sent2'].str.len().mean()), color='tab:red', linestyle='dashed')
axis_1.text(0.722, 0.3, sent1_sent2_string, transform=axis_1.transAxes, bbox=dict(boxstyle='square', facecolor='tab:green', edgecolor='tab:green', alpha=0.1))
# axis_1.text(0.862, 0.76, sent1_sent2_string, transform=axis_1.transAxes, bbox=dict(boxstyle='square', facecolor='tab:green', edgecolor='tab:green', alpha=0.1)) # figsize = (8, 8)
axis_2.set_title('ARG1+ARG2 (with context)', fontweight='bold')
axis_2.set_xlabel('Character length')
axis_2.set_xlim((0, 3500))
axis_2.set_xticks([0, 700, 1400, 2100, 2800, 3500])
axis_2.set_ylim((0, 1000))
axis_2.hist(df['arg1_arg2'].str.len(), bins=50, color='tab:orange')
axis_2.axvline(round(df['arg1_arg2'].str.len().mean()), color='tab:red', linestyle='dashed')
axis_2.text(0.69, 0.3, arg1_arg2_string, transform=axis_2.transAxes, bbox=dict(boxstyle='square', facecolor='tab:orange', edgecolor='tab:orange', alpha=0.1))
# axis_2.text(0.846, 0.76, arg1_arg2_string, transform=axis_2.transAxes, bbox=dict(boxstyle='square', facecolor='tab:orange', edgecolor='tab:orange', alpha=0.1)) # figsize = (8, 8)
plt.tight_layout()
figure.savefig('character_length_histogram.png', format='png', dpi=2400, bbox_inches="tight")
'''

# convert 'majoritylabel_sampled' column to numerical values
encoder = LabelEncoder()
encoder.fit(df['majoritylabel_sampled'])
df['majoritylabel_sampled'] = encoder.transform(df['majoritylabel_sampled'])

# split data into training/testing (80/20) with the same distribution of labels in 'majoritylabel_sampled'
gs_test = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gs_test.split(df, df['majoritylabel_sampled']))
df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]

# convert 'majoritylabel_sampled' column back to original values
df_train['majoritylabel_sampled'] = encoder.inverse_transform(df_train['majoritylabel_sampled'])
df_test['majoritylabel_sampled'] = encoder.inverse_transform(df_test['majoritylabel_sampled'])

'''
# plot data distribution for the training/testing datasets
figure, (axis_1, axis_2) = plt.subplots(2, 1, figsize=(3, 5))
axis_1.set_title(f'Training', fontweight='bold')
axis_1.tick_params(axis='x', rotation=90)
axis_1.set_ylim((0, 1500))
axis_1.set_yticks([0, 750, 1500]) 
axis_1.bar(df_train['majoritylabel_sampled'].value_counts().index, df_train['majoritylabel_sampled'].value_counts(), color='tab:green')
# axis_1.text(0.665, 0.905, f'{len(df_train)} instances ({len(df_train)/(len(df_train)+len(df_test))*100:.2f}%)', transform=axis_1.transAxes, bbox=dict(boxstyle='square', facecolor='tab:green', edgecolor='tab:green', alpha=0.1)) # figsize = (8, 8)
axis_2.set_title(f'Testing', fontweight='bold')
axis_2.tick_params(axis='x', rotation=90)
axis_2.set_ylim((0, 400))
axis_2.set_yticks([0, 200, 400]) 
axis_2.bar(df_test['majoritylabel_sampled'].value_counts().index, df_test['majoritylabel_sampled'].value_counts(), color='tab:orange')
# axis_2.text(0.665, 0.905, f'{len(df_test)} instances ({len(df_test)/(len(df_train)+len(df_test))*100:.2f}%)', transform=axis_2.transAxes, bbox=dict(boxstyle='square', facecolor='tab:orange', edgecolor='tab:orange', alpha=0.1)) # figsize = (8, 8)
plt.tight_layout()
figure.savefig('discogem_data_distribution.png', format='png', dpi=2400, bbox_inches="tight")
'''

# initialize tokenizer for RoBERTa-base
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# prepare features and targets
x_train_sent = tokenizer(list(df_train['sent1_sent2'].copy()), truncation=True, padding=True, return_tensors='pt')
x_train_arg  = tokenizer(list(df_train['arg1_arg2']),   truncation=True, padding=True, return_tensors='pt')
x_test_sent  = tokenizer(list(df_test['sent1_sent2'].copy()),  truncation=True, padding=True, return_tensors='pt')
x_test_arg   = tokenizer(list(df_test['arg1_arg2']),    truncation=True, padding=True, return_tensors='pt')
y_train      = df_train.iloc[:,8:].copy()
y_test       = df_test.iloc[:,8:].copy()

# generate PyTorch datasets
train_sent_dataset = IDRR_Dataset(x_train_sent, y_train)
train_arg_dataset  = IDRR_Dataset(x_train_arg,  y_train)
test_sent_dataset  = IDRR_Dataset(x_test_sent,  y_test)
test_arg_dataset   = IDRR_Dataset(x_test_arg,   y_test)

# generate PyTorch dataloaders
train_sent_loader = DataLoader(train_sent_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=THREADS)
train_arg_loader  = DataLoader(train_arg_dataset,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=THREADS)
test_sent_loader  = DataLoader(test_sent_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=THREADS)
test_arg_loader   = DataLoader(test_arg_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=THREADS)

# initalize classification model for Camembert-Base with a single linear classification layer
num_of_classes = len(y_train.iloc[0])
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_of_classes)
model.to(DEVICE)

# define optimizer and loss functions
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98))
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch:(1-DECAY))
loss_fn   = torch.nn.CrossEntropyLoss(reduction='mean')

print('\n')
print('Starting training...')
start_time = time.time()

# train the model
model.train()

for epoch in range(EPOCHS):  
    for batch_index, batch in enumerate(train_arg_loader):

        # prepare data
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs['logits'], labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        if not batch_index % 10:
            # get majority label classes
            true_majority_labels = torch.argmax(labels, dim=1).numpy()
            pred_majority_labels = torch.argmax(outputs['logits'], dim=1).numpy()

            # calculate metrics
            f1_score    = metrics.f1_score(true_majority_labels, pred_majority_labels, average='weighted', zero_division=0)
            precision   = metrics.precision_score(true_majority_labels, pred_majority_labels, average='weighted', zero_division=0)
            recall      = metrics.recall_score(true_majority_labels, pred_majority_labels, average='weighted', zero_division=0)
            
            print(f'Epoch:      {epoch+1:04d}/{EPOCHS:04d} | '
                  f'Batch:      {batch_index:04d}/{len(train_arg_loader):04d} | '
                  f'Learn Rate: {optimizer.param_groups[0]["lr"]:.8f} | '
                  f'Loss:       {loss:.4f} | '
                  f'F1 Score:   {f1_score:.4f} | '
                  f'Precision:  {precision:.4f} | '
                  f'Recall:     {recall:.4f} ')
            
            wandb.log({'Loss':      loss,
                       'F1 Score':  f1_score,
                       'Precision': precision,
                       'Recall':    recall})
    
    scheduler.step()

print(f'Total training time: {(time.time()-start_time)/60:.2f} minutes')

# create model folder
if not os.path.exists('Model-Arg-'+str(LR)+'-'+str(DECAY)):
    os.makedirs('Model-Arg-'+str(LR)+'-'+str(DECAY))

# save model configuration
torch.save(model.state_dict(), 'Model-Arg-'+str(LR)+'-'+str(DECAY)+'/'+MODEL_NAME+'-arg.pt')

print('\n')
print('Starting inference...')
start_time = time.time()

# test the model
model.eval()

with torch.no_grad():
    all_true_labels = []
    all_pred_labels = []
    all_true_majority_labels = []
    all_pred_majority_labels = []

    for batch_index, batch in enumerate(test_arg_loader):

        # prepare data
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # get soft label classes
        all_true_labels.extend(labels.tolist())
        all_pred_labels.extend(outputs['logits'].tolist())
        # get majority label classes
        all_true_majority_labels.extend(torch.argmax(labels, dim=1).tolist())
        all_pred_majority_labels.extend(torch.argmax(outputs['logits'], dim=1).tolist())

        # logging
        if not batch_index % 100:
            print(f'Test Batch: {batch_index:04d}/{len(test_arg_loader):04d}')

print(f'Total inference time: {(time.time()-start_time)/60:.2f} minutes')
    
# save labels to csv files
label_names = list(df_test.iloc[:,8:].columns)
with open('Model-Arg-'+str(LR)+'-'+str(DECAY)+'/all_true_labels-arg.csv', 'w') as file:
    write = csv.writer(file)
    write.writerow(label_names)
    write.writerows(all_true_labels)
with open('Model-Arg-'+str(LR)+'-'+str(DECAY)+'/all_pred_labels-arg.csv', 'w') as file:
    write = csv.writer(file)
    write.writerow(label_names)
    write.writerows(all_pred_labels)
with open('Model-Arg-'+str(LR)+'-'+str(DECAY)+'/all_true_majority_labels-arg.csv', 'w') as file:
    write = csv.writer(file)
    write.writerow(all_true_majority_labels)
with open('Model-Arg-'+str(LR)+'-'+str(DECAY)+'/all_pred_majority_labels-arg.csv', 'w') as file:
    write = csv.writer(file)
    write.writerow(all_pred_majority_labels)

# calculate metrics
f1_score    = metrics.f1_score(all_true_majority_labels, all_pred_majority_labels, average='weighted', zero_division=0)
precision   = metrics.precision_score(all_true_majority_labels, all_pred_majority_labels, average='weighted', zero_division=0)
recall      = metrics.recall_score(all_true_majority_labels, all_pred_majority_labels, average='weighted', zero_division=0)

# logging
print(f'F1 Score:   {f1_score:.4f} | '
      f'Precision:  {precision:.4f} | '
      f'Recall:     {recall:.4f}')

# calculate confusion matrix
confusion_matrix = metrics.confusion_matrix(all_true_majority_labels, all_pred_majority_labels, labels=np.arange(num_of_classes, dtype=int))

# plot confusion matrix
figure, axis = plt.subplots(1, 1, figsize=(16, 16))
axis.matshow(confusion_matrix, cmap='Greens', alpha=0.4)
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        axis.text(x=j, y=i, s=confusion_matrix[i, j], va='center', ha='center', fontsize=11)
axis.set_xticks(ticks=np.arange(num_of_classes), labels=label_names, rotation='vertical')
axis.set_yticks(ticks=np.arange(num_of_classes), labels=label_names)
axis.set_xlabel('Predictions', fontweight='bold')
axis.set_ylabel('Targets', fontweight='bold')
axis.xaxis.tick_bottom()
plt.tight_layout()
figure.savefig('Model-Arg-'+str(LR)+'-'+str(DECAY)+'/confusion_matrix-arg.png', format='png', dpi=300, bbox_inches="tight")