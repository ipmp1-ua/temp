import json
import torch
from data import GrandStaffDataset
from smt_trainer import SMT_Trainer

from ExperimentConfig import experiment_config_from_dict
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
import cv2
from data_augmentation.data_augmentation import convert_img_to_tensor
from smt_model import SMTModelForCausalLM
import os
from weight_transfer import map_lm_to_smt

torch.set_float32_matmul_precision('medium')


with open("config/GrandStaff/grandstaff.json", "r") as f:
  config = experiment_config_from_dict(json.load(f))
datamodule = GrandStaffDataset(config=config.data)

smt = SMT_Trainer.load_from_checkpoint("/workspace/weights/GrandStaff/wow_best_de_verdad_12.ckpt")


wandb_logger = WandbLogger(project='SMT_Reimplementation', group="GrandStaff", name="SMTest", log_model=False)

try:
  early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01, patience=5, mode="min", verbose=True)

  checkpointer = ModelCheckpoint(dirpath="weights/GrandStaff/", filename="GrandStaff_SMT_nianga-{epoch}", 
                                every_n_epochs=1, verbose=True)

  trainer = Trainer(max_epochs=15,
                    check_val_every_n_epoch=1, 
                    logger=wandb_logger, callbacks=[checkpointer, early_stopping], precision='16-mixed')

  trainer.test(smt, datamodule=datamodule)
except KeyboardInterrupt:
  print("Training interrupted. Saving the model...")
  trainer.save_checkpoint("weights/GrandStaff/GrandStaff_SMT_nianga-interrupt.ckpt")
  wandb_logger.experiment.finish()
  print("Model saved.")