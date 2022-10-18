from typing import Optional
from pprint import pprint

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import AdamW, AutoConfig, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput
from torchmetrics import MetricCollection, Recall, Precision, Accuracy, ConfusionMatrix

from job_offers_classifier.classification_utils import *

class FullyConnectedOutput(nn.Module):
    def __init__(self, input_size, output_size, layer_units=(10,), nonlin=nn.ReLU(), hidden_dropout=0, output_nonlin=nn.Softmax(dim=1), criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nonlin = nonlin
        self.layer_units = layer_units
        self.output_nonlin = output_nonlin
        self.criterion = criterion
        self.hidden_dropout = hidden_dropout

        sequence = []
        units = [self.input_size] + list(self.layer_units) + [self.output_size]
        for in_size, out_size in zip(units, units[1:]):
            sequence.extend([nn.Linear(in_size, out_size), self.nonlin, nn.Dropout(self.hidden_dropout)])

        sequence = sequence[:-2]
        self.sequential = nn.Sequential(*sequence)

    def forward(self, batch, labels=None):
        output = self.sequential(batch)

        if labels is not None:
            return self.criterion(output, labels), output
        else:
            return self.output_nonlin(output)


class TransformerClassifier(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        output_type: str = "linear",
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: float = 50,  # from 0-1 for % of training steps, >1 for number of steps
        weight_decay: float = 0.01,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        hidden_dropout: float = 0.0,
        eval_top_k: int = 10,
        freeze_transformer: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        if self.hparams.verbose:
            print(f"Initializing TransformerClassifier with model_name={model_name_or_path}, output_type={output_type}, num_labels={num_labels}, learning_rate={learning_rate}, weight_decay={weight_decay}, warmup_steps={warmup_steps} ...")

        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            finetuning_task=None
        )
        self.transformer = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.output = self._get_output_layer(output_type, num_labels, hidden_dropout)

        metric_dict = {
            "acc/r@1": Accuracy(num_classes=num_labels),
            "macro_acc": Accuracy(num_classes=num_labels, average='macro'),
            #"cf_matrix_true": ConfusionMatrix(num_classes=num_labels, normalize='all'),
            #"cf_matrix_all": ConfusionMatrix(num_classes=num_labels, normalize='all')
        }

        for i in range(2, min(num_labels, eval_top_k + 1)):
            metric_dict[f"r@{i}"] = Recall(num_classes=num_labels, top_k=i)

        self.metrics = MetricCollection(metric_dict)


    def _get_output_layer(self, output_type, output_size, hidden_dropout):
        if output_type == "linear":
            return FullyConnectedOutput(self.config.hidden_size, output_size, layer_units=(), hidden_dropout=hidden_dropout, output_nonlin=nn.Softmax(dim=1), criterion=nn.CrossEntropyLoss())
        elif output_type == "nn":
            return FullyConnectedOutput(self.config.hidden_size, output_size, layer_units=(self.config.hidden_size,), hidden_dropout=hidden_dropout, output_nonlin=nn.Softmax(dim=1), criterion=nn.CrossEntropyLoss())
        else:
            raise ValueError("Unknown output_type for TransformersClassifier")

    def forward(self, batch, labels=None):
        transformer_output = self.transformer(batch['input_ids'], attention_mask=batch['attention_mask'])
        transformer_output = transformer_output.last_hidden_state[:, 0, :]
        if self.output is None:
            return transformer_output
        else:
            return self.output.forward(transformer_output, labels=labels)

    def training_step(self, batch, batch_idx):
        if batch['labels'] is not None:  # Windows fix
            batch['labels'] = batch['labels'].type(torch.LongTensor)
            batch['labels'] = batch['labels'].to(self.device)
    
        loss, scores = self.forward(batch, batch['labels'])
        self.log('train_loss', loss, on_epoch=True, logger=True)
        return loss

    def _eval_step(self, batch, eval_name='val'):
        if batch['labels'] is not None:  # Windows fix
            batch['labels'] = batch['labels'].type(torch.LongTensor)
            batch['labels'] = batch['labels'].to(self.device)
    
        loss, scores = self.forward(batch, batch['labels'])
        self.log(f'{eval_name}_performance', self.metrics(scores, batch['labels']), on_epoch=True, logger=True)
        self.log(f'{eval_name}_loss', loss, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, eval_name='val')

    def validation_epoch_end(self, outputs):
        if self.hparams.verbose:
            print("Validation performance:")
            pprint(self.metrics.compute())
        self.metrics.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, eval_name='test')

    def test_epoch_end(self, outputs):
        if self.hparams.verbose:
            print("Test performance:")
            pprint(self.metrics.compute())

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return

        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches
        dl_size = len(train_loader.dataset) * self.trainer.max_epochs
        self.total_steps = dl_size // tb_size // ab_size

        self.num_warmup_steps = self.hparams.warmup_steps
        if self.hparams.warmup_steps < 1:
            self.num_warmup_steps = int(self.total_steps * self.hparams.warmup_steps)

        if self.hparams.verbose:
            print("Warmup_steps:", self.num_warmup_steps)
            print("Total steps:", self.total_steps)

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = AdamW(self._get_optimizer_grouped_parameters(),
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def save_transformer(self, ckpt_dir):
        self.transformer.save_pretrained(ckpt_dir)

    def _get_optimizer_grouped_parameters(self, layer_wise_lr=False, layer_wise_lr_mutli=1.1):
        # It is suggested to not use any decay for bias, LayerNorm.weight and LayerNorm.weight layers.
        no_decay = ["bias", "LayerNorm.weight"]

        if layer_wise_lr:
            optimizer_grouped_parameters = []
            for name, params in self.named_parameters():
                weight_decay = 0.0 if any(nd in name for nd in no_decay) else self.hparams.weight_decay
                learning_rate = self.hparams.learning_rate

                if 'embeddings' in name or 'encoder' in name:
                    learning_rate /= 10

                    for i in range(0, 20):
                        if f'layer.{i}' in name:
                            learning_rate *= layer_wise_lr_mutli ** (i + 1)

                print(name, learning_rate)
                optimizer_grouped_parameters.append({
                    "params": params,
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                })

        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        return optimizer_grouped_parameters
