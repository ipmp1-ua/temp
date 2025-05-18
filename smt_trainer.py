import torch
import wandb
import random
import numpy as np
import torch.nn as nn
import lightning.pytorch as L
import tqdm

from torchinfo import summary
from eval_functions import compute_poliphony_metrics
from smt_model import SMTConfig
from smt_model import SMTModelForCausalLM



class SMT_Trainer(L.LightningModule):
    def __init__(self, maxh, maxw, maxlen, out_categories, padding_token, in_channels, w2i, i2w, d_model=256, dim_ff=256, num_dec_layers=8):
        super().__init__()
        self.config = SMTConfig(maxh=maxh, maxw=maxw, maxlen=maxlen, out_categories=out_categories,
                           padding_token=padding_token, in_channels=in_channels, 
                           w2i=w2i, i2w=i2w,
                           d_model=d_model, dim_ff=dim_ff, attn_heads=4, num_dec_layers=num_dec_layers, 
                           use_flash_attn=True)
        self.model = SMTModelForCausalLM(self.config)
        self.padding_token = padding_token
        
        self.preds = []
        self.grtrs = []

        self.val_count = 0
        self.running_cer = 0.0
        self.running_ser = 0.0
        self.running_ler = 0.0
        
        self.save_hyperparameters()
        
        summary(self, input_size=[(1,1,self.config.maxh,self.config.maxw), (1,self.config.maxlen)], 
                dtypes=[torch.float, torch.long])
        
    
    def configure_optimizers(self):
        return torch.optim.Adam(list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()), lr=1e-4, amsgrad=False)
    
    def forward(self, input, last_preds):
        return self.model(input, last_preds)
    
    def training_step(self, batch):
        x, di, y = batch
        outputs = self.model(encoder_input=x, decoder_input=di[:, :-1], labels=y)
        loss = outputs.loss
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        
        return loss
        

    def validation_step(self, val_batch):
        x, _, y = val_batch
        predicted_sequence, _ = self.model.predict(input=x)
        
        dec = "".join(predicted_sequence)

        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")
        print(["DEC"])
        print(dec.replace("**ekern_1.0","**kern"))

        gt = "".join([self.model.i2w[token.item()] for token in y.squeeze(0)[:-1]])
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")
        print(["GT"])

        print(gt.replace("**ekern_1.0","**kern"))

        gt = "".join([self.model.i2w[token.item()] for token in y.squeeze(0)])
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")
        print(["GT2"])

        print(gt.replace("**ekern_1.0","**kern"))

        self.preds.append(dec)
        self.grtrs.append(gt)
        cer, ser, ler = compute_poliphony_metrics([dec], [gt])

        self.val_count += 1
        n = self.val_count

        self.running_cer = (self.running_cer * (n - 1) + cer) / n
        self.running_ser = (self.running_ser * (n - 1) + ser) / n
        self.running_ler = (self.running_ler * (n - 1) + ler) / n

        print(f"[batch {n}] CER: {cer:.4f}  →  running_CER: {self.running_cer:.4f}")
        print(f"[batch {n}] SER: {ser:.4f}  →  running_SER: {self.running_ser:.4f}")
        print(f"[batch {n}] LER: {ler:.4f}  →  running_LER: {self.running_ler:.4f}")

        # registramos la métrica acumulada en el logger
        self.log("val_CER_running", self.running_cer, on_epoch=True, prog_bar=True)
        self.log("val_SER_running", self.running_ser, on_epoch=True, prog_bar=True)
        self.log("val_LER_running", self.running_ler, on_epoch=True, prog_bar=True)
        
    def on_validation_epoch_end(self, metric_name="val") -> None:
        cer, ser, ler = compute_poliphony_metrics(self.preds, self.grtrs)
        self.val_count = 1
        random_index = random.randint(0, len(self.preds)-1)
        predtoshow = self.preds[random_index]
        gttoshow = self.grtrs[random_index]
        print(f"[Prediction] - {predtoshow}")
        print(f"[GT] - {gttoshow}")
        print(f"[CER] - {cer}")
        print(f"[SER] - {ser}")
        print(f"[LER] - {ler}")
        self.log(f'{metric_name}_CER', cer, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_SER', ser, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_LER', ler, on_epoch=True, prog_bar=True)
        
        self.preds = []
        self.grtrs = []
        
        return ser
    
    def test_step(self, test_batch):
        return self.validation_step(test_batch)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end("test")