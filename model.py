import argparse
import logging
import os
from typing import DefaultDict
import numpy as np
import pandas as pd
from pytorch_lightning.core.hooks import CheckpointHooks
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import CustumDataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
DEFAULT_PATH = ""
class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default=DEFAULT_PATH + 'train_original.json',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default=DEFAULT_PATH +'train_original.json',
                            help='test file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=2,
                            help='')
        
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        return parser

class KobartSummary(pl.LightningDataModule):
    
    def __init__(self,train_file,test_file,tokenizer,
                max_len = 512,
                batch_size = 8,
                num_workers = 5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        
        if tokenizer is None:
            self.tok = get_kobart_tokenizer()
        else:
            self.tok = tokenizer
            
        self.num_workers = num_workers
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
        parents = [parent_parser],add_help = False)
        
        parser.add_argument("--num_workers",
                           type = int,
                           default = 5,
                           help = 'num of worker for dataloader')
        
        return parser
    
    def setup(self,stage):
        self.train = CustumDataset(self.train_file_path,
                                  self.tok,
                                  self.max_len,
                                  pad_idx = 0)
        
        self.test = CustumDataset(self.test_file_path,
                                 self.tok,
                                 self.max_len,
                                 pad_idx = 0)
    
    def train_dataloader(self):
        
        train = DataLoader(self.train,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers, shuffle = True)
        return train
    
    def val_dataloader(self):
        
        val = DataLoader(self.test,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers, shuffle = False)
        return val
    
    def test_dataloader(self):
        
        test = DataLoader(self.test,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers, shuffle = False)
        return test

class Base(pl.LightningModule):
    def __init__(self,hparams,**kwargs) -> None:
        super(Base,self).__init__()
        self.save_hyperparameters(hparams)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=4,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')

        return parser

    
    def configure_optimizers(self):
        #prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
        #무슨 역할인지 잘 모르겠음.
        optimizer_grouped_parameters = [
            {"params" : [p for n,p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay' : 0.01},
            {"params" : [p for n,p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay' : 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters,
                         lr = self.hparams.lr, correct_bias = False)
        
        num_workers = self.hparams.num_workers
        
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')

        num_train_steps = 2#int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')

        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer],[lr_scheduler]


class KoBartConditionalGeneration(Base):
    
    def __init__(self,hparams,**kwargs):
        super(KoBartConditionalGeneration,self).__init__(hparams,**kwargs)
        self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.tokenizer = get_kobart_tokenizer()
        
    
    def forward(self,inputs):
        
        attention_mask = inputs['encoder_input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float().to('cuda')
        
        return self.model(input_ids = inputs["encoder_input_ids"],
                         attention_mask = attention_mask,
                         decoder_input_ids = inputs['decoder_input_ids'],
                          decoder_attention_mask = decoder_attention_mask,
                          labels = inputs['label_ids'],return_dict = True)
    
    def training_step(self,batch,batch_idx):
        out = self(batch)
        loss = out.loss
        self.log('train_loss',loss,prog_bar = True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        out = self(batch)
        loss = out['loss']
        return (loss)
    
    def validation_epoch_end(self,outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        
        self.log("val_loss",torch.stack(losses).mean(),prog_bar = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KoBART Summarization')

    parser.add_argument('--checkpoint_path',
                        type=str,
                        help='checkpoint path')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KobartSummary.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    logging.info(args)

    model = KoBartConditionalGeneration(args)

    dm = KobartSummary(args.train_file,
    args.test_file,None,max_len =512,batch_size= args.batch_size,
    num_workers= args.num_workers)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                        dirpath=args.default_root_dir,
                                                        filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                        verbose=True,
                                                        save_last=True,
                                                        mode='min',
                                                        save_top_k=-1)
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(DEFAULT_PATH, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                                callbacks=[checkpoint_callback, lr_logger],
                                                gpus=-1,auto_select_gpus = False)
    trainer.fit(model,dm)