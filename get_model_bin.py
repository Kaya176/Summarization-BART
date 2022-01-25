import argparse
from model import KoBartConditionalGeneration
from transformers.models.bart import BartForConditionalGeneration
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default="./summary/hparams.yaml", type=str)
parser.add_argument("--model_binary", default="./summary/summary_model.ckpt", type=str)
parser.add_argument("--output_dir", default='kobart_summary', type=str)
args = parser.parse_args()

with open(args.hparams) as f:
    hparams = yaml.load(f)
    
inf = KoBartConditionalGeneration.load_from_checkpoint(args.model_binary, hparams=hparams)

inf.model.save_pretrained(args.output_dir)