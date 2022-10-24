import torch
import torch.nn as nn
from .victim import Victim
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence
from torchinfo import summary


class PLMVictim(Victim):
    """
    PLM victims. Support Huggingface's Transformers.

    Args:
        device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
        model (:obj:`str`, optional): The model to use. Defaults to "bert".
        path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
        num_classes (:obj:`int`, optional): The number of classes. Defaults to 2.
        max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 512.
    """
    def __init__(
            self, 
            device: Optional[str] = "gpu",
            model: Optional[str] = "bert",
            path: Optional[str] = "bert-base-uncased",
            num_classes: Optional[int] = 2,
            max_len: Optional[int] = 512,
            freeze_features: Optional[bool] = None,
            unfreeze_n_top: Optional[int] = None,
            teacher_name: Optional[str] = None,
            kd_temperature: Optional[float] = None,
            kd_alpha: Optional[float] = None,
            use_adapter: Optional[bool] = None,
            **kwargs
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes
        # you can change huggingface model_config here
        self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config)

        # for fine-tuning top-layers only
        if freeze_features:
            if unfreeze_n_top is not None:
                unfreeze_layers = list(range(self.model_config.num_hidden_layers))[-1*unfreeze_n_top:]
                unfreeze_layer_rules = [f'.{x}.' for x in unfreeze_layers]
            else:
                unfreeze_layer_rules = []
            for n, p in self.plm.named_parameters():
                if 'classifier' not in n and not any([x in n for x in unfreeze_layer_rules]):
                    p.requires_grad = False
                else:
                    print(n, p.requires_grad)

        # for knowledge distillation
        self.kd_temperature = kd_temperature
        self.kd_alpha = kd_alpha        
        if teacher_name:
            self.teacher = AutoModelForSequenceClassification.from_pretrained(teacher_name)
            assert self.kd_temperature is not None and self.kd_alpha is not None
        else:
            self.teacher = None

        # for adapter tuning
        # needs a separate environment with transformers replaced by adapter-transformers
        if use_adapter:
            from transformers import AdapterConfig
            task_name = 'sst2'
            adapter_config = AdapterConfig.load(
                'pfeiffer',
                reduction_factor=4,
            )
            self.plm.add_adapter(task_name, config=adapter_config)
            self.plm.train_adapter([task_name])
            print(self.plm)

        print(summary(self.plm))
        
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.to(self.device)
        
    def to(self, device):
        self.plm = self.plm.to(device)
        if self.teacher:
            self.teacher = self.teacher.to(device)

    def forward(self, inputs):
        output = self.plm(**inputs, output_hidden_states=True)
        return output

    def get_repr_embeddings(self, inputs):
        output = self.plm.getattr(self.model_name)(**inputs) # batch_size, max_len, 768(1024)
        return output[:, 0, :]


    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels 
    
    @property
    def word_embedding(self):
        head_name = [n for n,c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight
    
