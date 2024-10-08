import torch
import streamlit as st
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration

def load_model():
    model = BartForConditionalGeneration.from_pretrained("./kobart_summary/")
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()
st.title("KoBART 요약 test")
text = st.text_area("뉴스 입력: ")

st.markdown("## 원문")
st.write(text)

if text:
    text = text.replace("\n","")
    st.markdown("## KoBart Model Summarization")
    with st.spinner("processing.."):
        input_ids = tokenizer.encode(text)
        input_ids = torch.Tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids,eos_token_id=1,max_length=512,num_beams=5)
        output = tokenizer.decode(output[0],skip_special_tokens = True)
    st.write(output)