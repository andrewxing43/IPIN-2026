# model.py
import torch
from torch import nn
from utils import GradReverse

class FeatureExtractor(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2), nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2), nn.Dropout2d(0.5),
           
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*43*43, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1, True),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1, True)
        )
    def forward(self, x): return self.head(self.net(x))  # (B,128)

class RegressorEnsemble(nn.Module):
    def __init__(self, in_dim=128, out_dim=2):
        super().__init__()
        def head():
            return nn.Sequential(
                nn.Linear(in_dim, 128), nn.LeakyReLU(0.1, True),
                nn.Linear(128, 64), nn.LeakyReLU(0.1, True),
                nn.Linear(64, out_dim)
            )
        self.m1 = head(); self.m2 = head()
        self.m3 = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LeakyReLU(0.1, True),
            nn.Linear(128, 128), nn.LeakyReLU(0.1, True),
            nn.Linear(128, out_dim)
        )
    def forward(self, feats):
        y1 = self.m1(feats); y2 = self.m2(feats); y3 = self.m3(feats)
        y  = (y1 + y2 + y3) / 3.0
        return y, (y1, y2, y3)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=128, n_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LeakyReLU(0.1, True), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.LeakyReLU(0.1, True), nn.Dropout(0.5),
            nn.Linear(64, n_domains)
        )
    def forward(self, feats): return self.net(feats)

class DANNModel(nn.Module):
    def __init__(self, in_ch=1, feat_dim=128, grl_lambda=0.0):
        super().__init__()
        self.extractor = FeatureExtractor(in_ch=in_ch)
        self.regressor = RegressorEnsemble(in_dim=feat_dim, out_dim=2)
        self.domain_dis = DomainDiscriminator(in_dim=feat_dim, n_domains=2)
        self.grl = GradReverse(grl_lambda)

    def forward(self, x, grl_lambda=None):
        feats = self.extractor(x)
        y, _ = self.regressor(feats)
        if grl_lambda is not None: self.grl.lambd = grl_lambda
        dom_logits = self.domain_dis(self.grl(feats))
        return y, dom_logits

    @torch.no_grad()
    def predict_xy(self, x):
        feats = self.extractor(x); y, _ = self.regressor(feats); return y

    @torch.no_grad()
    def predict_heads(self, x):
        feats = self.extractor(x)
        y1 = self.regressor.m1(feats)
        y2 = self.regressor.m2(feats)
        y3 = self.regressor.m3(feats)
        avg = (y1 + y2 + y3) / 3.0
        return avg, (y1, y2, y3)
