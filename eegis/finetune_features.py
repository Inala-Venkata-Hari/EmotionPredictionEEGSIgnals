"""Training-only adaptation of LaBraM's last block; frozen earlier blocks."""
import copy
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
import torch
from torch import nn

from .accuracy_features import validate
from .pretrained_features import encoder,patches


def prefix_tokens(x):
    model,device,channels=encoder();x=validate(x);values=[];torch.set_num_threads(4)
    with torch.inference_mode():
        for start in range(0,len(x),16):
            inputs=torch.from_numpy(patches(x[start:start+16],'normalized')).to(device)
            batch,n,time,_=inputs.shape
            tokens=model.patch_embed(inputs)
            tokens=torch.cat([model.cls_token.expand(batch,-1,-1),tokens],dim=1)
            position=model.pos_embed[:,channels]
            spatial=position[:,1:,:].unsqueeze(2).expand(batch,-1,time,-1).flatten(1,2)
            tokens+=torch.cat([position[:,:1].expand(batch,-1,-1),spatial],dim=1)
            temporal=model.time_embed[:,:time].unsqueeze(1).expand(batch,n,-1,-1).flatten(1,2)
            tokens[:,1:]+=temporal
            tokens=model.pos_drop(tokens)
            for block in model.blocks[:-1]:tokens=block(tokens,rel_pos_bias=None)
            values.append(tokens.cpu().numpy())
    return np.concatenate(values)


class LastBlockHead(nn.Module):
    def __init__(self):
        super().__init__();model,_,_=encoder()
        self.block=copy.deepcopy(model.blocks[-1]).cpu()
        self.norm=copy.deepcopy(model.norm).cpu()
        self.dropout=nn.Dropout(.25);self.head=nn.Linear(200,2)
        self.requires_grad_(True)

    def forward(self,x):
        x=self.norm(self.block(x,rel_pos_bias=None))
        return self.head(self.dropout(x[:,1:].mean(dim=1)))


class LastBlockClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,epochs=5,seed=42):self.epochs=epochs;self.seed=seed

    def fit(self,x,y):
        torch.set_num_threads(4);torch.manual_seed(self.seed);torch.use_deterministic_algorithms(True)
        self.classes_,encoded=np.unique(y,return_inverse=True)
        if list(self.classes_)!=[6,8]:raise ValueError('Expected No/Sí')
        self.network_=LastBlockHead();self.training_device_='cuda' if torch.cuda.is_available() else 'cpu'
        device=torch.device(self.training_device_);self.network_.to(device)
        opt=torch.optim.AdamW([{'params':list(self.network_.block.parameters())+list(self.network_.norm.parameters()),'lr':1e-4},
                               {'params':self.network_.head.parameters(),'lr':1e-3}],weight_decay=.01)
        counts=np.bincount(encoded,minlength=2)
        loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(len(y)/(2*counts),dtype=torch.float32,device=device))
        xx=torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32);yy=torch.tensor(encoded,dtype=torch.long)
        rng=np.random.default_rng(self.seed);self.history_=[]
        for epoch in range(self.epochs):
            self.network_.train();total=0.;correct=0
            for ix in np.array_split(rng.permutation(len(xx)),max(1,int(np.ceil(len(xx)/32)))):
                a,b=xx[ix].to(device),yy[ix].to(device);opt.zero_grad(set_to_none=True)
                logits=self.network_(a);loss=loss_fn(logits,b);loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(),1.);opt.step()
                total+=float(loss.detach())*len(ix);correct+=int((logits.argmax(dim=1)==b).sum())
            self.history_.append({'epoch':epoch+1,'training_loss':total/len(xx),'training_accuracy':correct/len(xx)})
        self.network_.to('cpu');self.network_.eval();return self

    def predict(self,x):
        self.network_.eval();values=torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32)
        with torch.no_grad():p=torch.cat([self.network_(a).argmax(dim=1) for a in values.split(64)]).numpy()
        return self.classes_[p]


@dataclass
class FineTuneModel:
    estimator:object
    bank:str
    samples:int

    def features(self,x):
        x=validate(x)
        if x.shape[1]!=self.samples:raise ValueError('Input duration differs')
        return prefix_tokens(x)

    def predict(self,x):return self.estimator.predict(self.features(x))
