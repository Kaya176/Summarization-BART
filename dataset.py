#dataset
'''
dataset을 만들기 위해 구현해야할 것:
__init__ / __getitem__ / __len__

특히, __getitem__ 은 encoder_input_id, decoder_input_id,label_id 정보가 반드시 필요함.
encoder_input_id는 [1,2,3,4,5] 이런식 / 다른 정보도 마찬가지로 토큰화시키면 됨.
'''
from torch.utils.data import Dataset
import numpy as np
from load_data import load_data


class CustumDataset(Dataset):
    
    def __init__(self,file_name,tokenizer,max_len,pad_idx=None,ignore_idx = -100):
        self.tokenizer = tokenizer #토크나이저 설정
        self.max_len = max_len #sentence 최대길이 지정
        self.data = load_data(file_name) #데이터 불러오기
        self.len = self.data.shape[0] #데이터의 총 갯수
        self.pad_idx = pad_idx
        self.ignore_idx = ignore_idx
        
    def _add_padding2data(self,sentence):
        '''
        입력한 sequence가 max_len보다 작은 경우 padding을 진행해줌으로써 길이를 모두 동일하게 맞춰준다.
        반대로 sentence가 max_len보다 크거나 같은 경우, max_len만큼만 쓸 것이므로 잘라준다.
        '''
        if len(sentence) < self.max_len:
            pad_seq = np.array([self.pad_idx] * (self.max_len - len(sentence)))
            sentence = np.concatenate([sentence,pad_seq])
        else:
            sentence = sentence[:self.max_len]
        return sentence
    
    def _add_ignore2data(self,sentence):
        '''
        label(요약문)에 사용하는 함수로써 padding과 더불어 ignore_index를 추가해줌. 불필요한 길이를 방지하기 위함인거같음.(더 알아보기)
        '''
        if len(sentence) < self.max_len:
            pad_seq = np.array([self.ignore_idx] * (self.max_len - len(sentence)))
            sentence = np.concatenate([sentence,pad_seq])
        else:
            sentence = sentence[:self.max_len]
        return sentence
        
    def __getitem__(self,idx):
        #just for one line
        line = self.data.iloc[idx]
        sentence = line['sentence']
        abst = line['abstract']
        #encode : encoder input
        encoder_input_ids = self.tokenizer.encode_plus(sentence).input_ids
        encoder_input_ids = self._add_padding2data(encoder_input_ids)
        #encode : label (요약문)
        label_ids = self.tokenizer.encode_plus(abst).input_ids
        label_ids.append(self.tokenizer.eos_token_id)
        #encode : decoder input
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self._add_padding2data(dec_input_ids)
        #add ignore_idx
        label_ids = self._add_ignore2data(label_ids)
        return {"encoder_input_ids" : np.array(encoder_input_ids,dtype=np.int_) ,
                "decoder_input_ids" : np.array(dec_input_ids,dtype=np.int_),
                'label_ids' : np.array(label_ids,dtype=np.int_)}
    
    def __len__(self):
        return self.len