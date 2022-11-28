import captum

import spacy

import torch
import torchtext
import torchtext.data

import torch.nn as nn
import torch.nn.functional as F

from torchtext.vocab import Vocab

from captum.attr import LayerConductance, LayerIntegratedGradients, TokenReferenceBase, visualization

from transformers import AutoTokenizer, BertForSequenceClassification, AdapterConfig


for package in (captum, spacy, torch, torchtext):
    print(package.__name__, package.__version__)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
task_name = 'sst2'
adapter_config = AdapterConfig.load(
    'pfeiffer',
    reduction_factor=4,
)
model.add_adapter(task_name, config=adapter_config)
model.train_adapter([task_name])

ckpt = 'models/clean-badnets-0.1/1669591639/best.ckpt'
state_dict = torch.load(ckpt)
state_dict = {k.replace('plm.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
print(model)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

def predict(inputs):
    return model(inputs)[0]

def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

# # for negative attribution
def custom_forward(inputs):
    preds = predict(inputs)
    # use `torch.softmax(preds, dim = 1)[:, 1]` for positive attribution
    return torch.softmax(preds, dim = 1)[:, 0] 

lig = LayerIntegratedGradients(custom_forward, model.bert.embeddings)

def calculate_attribution(text):
    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id)
    score = predict(input_ids)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        n_steps=7000,
                                        internal_batch_size=5,
                                        return_convergence_delta=True)
    # print(attributions, delta)
    return all_tokens, score, attributions, delta

def summarize_attributions(attributions, is_neg=True):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    if is_neg:
        attributions = -1 * attributions
    return attributions


import pandas as pd
data = pd.read_csv('/local/diwu/nlp_attack/OpenBackdoor/poison_data/sst-2/0/badnets/test-poison.csv')

triggers = set(["cf", "mn", "bb", "tq"])

poisoned_data, clean_data = [], []
for i, row in data.iterrows():
    if i == 5:
        break
    _, cur_text, _, _ = row
    poisoned_data.append(cur_text)
    clean_data.append(' '.join([x for x in cur_text.split() if x not in triggers]))

from tqdm import tqdm

poisoned_results = []
for text in tqdm(poisoned_data):
    all_tokens, score, attributions, delta = calculate_attribution(text)
    attributions_sum = summarize_attributions(attributions)
    # storing couple samples in an array for visualization purposes
    score_vis = visualization.VisualizationDataRecord(attributions_sum,
                                                      torch.softmax(score, dim = 1)[0][0],
                                                      torch.argmax(torch.softmax(score, dim = 1)[0]),
                                                      1,
                                                      text,
                                                      attributions_sum.sum(),       
                                                      all_tokens,
                                                      delta)
    poisoned_results.append([text, all_tokens, score, attributions, delta])

torch.save(poisoned_results, '20221127_adapter_poisoned_results.pt')
    
clean_results = []
for text in tqdm(clean_data):
    all_tokens, score, attributions, delta = calculate_attribution(text)
    attributions_sum = summarize_attributions(attributions)
    # storing couple samples in an array for visualization purposes
    score_vis = visualization.VisualizationDataRecord(attributions_sum,
                                                      torch.softmax(score, dim = 1)[0][0],
                                                      torch.argmax(torch.softmax(score, dim = 1)[0]),
                                                      1,
                                                      text,
                                                      attributions_sum.sum(),       
                                                      all_tokens,
                                                      delta)
    clean_results.append([text, all_tokens, score.cpu(), attributions.cpu(), delta.cpu()])


torch.save(clean_results, '20221127_adapter_clean_results.pt')
