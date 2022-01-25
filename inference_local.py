import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration

def load_model():
    model = BartForConditionalGeneration.from_pretrained("./kobart_summary/")
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()
text = input("글 입력 : ")

input_ids = tokenizer.encode(text)
input_ids = torch.Tensor(input_ids).long()
input_ids = input_ids.unsqueeze(0)
output = model.generate(input_ids)
output = tokenizer.decode(output[0],skip_special_tokens = True)
print("-"*70)
print("KoBART 요약문")
print(output)