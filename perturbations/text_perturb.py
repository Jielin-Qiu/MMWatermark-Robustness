import os
import json
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
from eda import *
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2', cache_folder = './.huggingface_cache')
from styleformer import Styleformer
import torch
torch.set_num_threads(4)
import warnings
warnings.filterwarnings('ignore')
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

import random
random.seed(1234)

### CHARACTER LEVEL ###

## Keyboard Augmenter 
def perturb_KeyboardAug_json(sentence, ratio):
    
    char_ratio=0.05*ratio
    aug = nac.KeyboardAug(aug_word_p=char_ratio)
    aug_sentences = aug.augment(sentence)
    if aug_sentences==None:
        aug_sentences = sentence
                
    return aug_sentences    


## OCR Augmenter 
def perturb_OcrAug_json(sentence, ratio):
    
    char_ratio=0.05*ratio
    aug = nac.OcrAug(aug_word_p=char_ratio)    
    aug_sentences = aug.augment(sentence)
    if aug_sentences==None:
        aug_sentences = sentence
                
    return aug_sentences


## Random Augmenter
def perturb_RandomCharAug_json(sentence, action, ratio):
    
    char_ratio=0.05*ratio   
    aug = nac.RandomCharAug(action, aug_word_p=char_ratio)
    
    aug_sentences = aug.augment(sentence)
    if aug_sentences==None:
        aug_sentences = sentence
                
    return aug_sentences        


### WORD LEVEL ###

def eda_perturb(sentence, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug):
    
    times= 0 
    aug_sentences = eda(sentence, alpha_sr,alpha_ri,alpha_rs,p_rd,num_aug)
    
    return aug_sentences


PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']

def insert_punctuation_marks(sentence, punc_ratio):
    words = sentence.split(' ')
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1))
    qs = random.sample(range(0, len(words)), q)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)
    return new_line


def insert_punc(sentence,ratio):
    
    times= 0 

    aug_sentences = insert_punctuation_marks(sentence, ratio)
                
    return aug_sentences




### SENTENCE LEVEL ###

back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de', 
    to_model_name='facebook/wmt19-de-en'
)

def perturb_back_trans_json(sentence):
    
    aug_sentences = back_translation_aug.augment(sentence)
    if aug_sentences==None:
        aug_sentences = sentence
                
    return aug_sentences


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1234)

def text_style_perturb(sentence, style_value):
    tmp = sentence
    sf = Styleformer(style = style_value)
    times= 0 
    # dont do while loop and only take the aug_sentences 
    while times < 100:
        aug_sentences = sf.transfer(sentence)
        if aug_sentences==None:
            aug_sentences = tmp
        #print("aug_sentences", aug_sentences)
        embeddings_aug = model.encode(aug_sentences)
        embeddings_base = model.encode(tmp)
        similarity_score = float(util.cos_sim(embeddings_aug, embeddings_base))
        if similarity_score < 0.9: ### KEEP THRESHOLD AND THE NUMBER OF WHILE TIMES 
            times = times+1
            #print("perturb again")
        else:
            #print(similarity_score)
            break

                
    return aug_sentences