# eval_test.py — run from code/SNPbag/
import torch
import numpy as np
from dataset import load_plink, encode_genotypes, encode_snp_ids
from model import build_phenotype_regressor, load_pretrained_encoder, PhenotypeRegressor
from finetune_phenotype import classification_metrics, split_individuals

# Load data
bim, fam, geno = load_plink("TestData/NewSyn_100k_cpbayes")
geno = geno[:, :14000]
bim = bim.iloc[:14000]
geno_tokens = torch.from_numpy(encode_genotypes(geno)).long()
snp_ids = torch.from_numpy(encode_snp_ids(bim)).long()

# Load phenotype from fam (2=case->1, 1=control->0)
import pandas as pd
fam_df = pd.read_csv("TestData/NewSyn_100k_cpbayes.fam", sep=r"\s+", header=None)
pheno = torch.where(torch.tensor(fam_df[5].values) == 2, torch.ones(len(fam_df)), torch.zeros(len(fam_df))).float()

# Split
split = split_individuals(len(pheno), val_fraction=0.10, test_fraction=0.20, seed=42)
test_idx = split["test"]

# Load checkpoint
ckpt = torch.load("checkpoints/finetune_cpbayes/linear-frozen-seed42/best.pt", map_location="cpu", weights_only=False)
model = build_phenotype_regressor(num_geno_tokens=5, num_snp_ids=14000, d_model=512, n_layers=16, n_heads=16, d_ff=2048)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Predict on test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

all_prob = []
all_true = []
batch_size = 1

with torch.no_grad():
    for start in range(0, len(test_idx), batch_size):
        idx = test_idx[start:start+batch_size]
        x_geno = geno_tokens[idx].to(device)
        x_snp = snp_ids.unsqueeze(0).expand(len(idx), -1).to(device)
        logits = model(x_geno, x_snp).squeeze(-1)
        probs = torch.sigmoid(logits)
        all_prob.append(probs.cpu())
        all_true.append(pheno[idx])
        print(f"  {min(start+batch_size, len(test_idx))}/{len(test_idx)}")

all_prob = torch.cat(all_prob).numpy()
all_true = torch.cat(all_true).numpy()

metrics = classification_metrics(all_true, all_prob)
print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
print(f"Test AUC:      {metrics['auc']:.4f}")