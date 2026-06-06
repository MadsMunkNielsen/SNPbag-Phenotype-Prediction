set -euo pipefail

DATA_DIR="SNPbag/TestData"
PLINK_PREFIX="${DATA_DIR}/NewSyn_100k"
MAX_SNPS=14000
PRETRAINED_CKPT="SNPbag/checkpoints/pretrain/snpbag_n100000_snps14000_seed42_best.pt"

echo "=========================================="
echo "Step 1: Simulate phenotype (CPBayes model)"
echo "=========================================="
Rscript simulate_pheno_cpbayes.R

echo ""
echo "=========================================="
echo "Step 2: Extract first ${MAX_SNPS} SNP IDs"
echo "=========================================="
head -n ${MAX_SNPS} "${PLINK_PREFIX}.bim" | awk '{print $2}' \
  > "${DATA_DIR}/first_14k_snps.txt"
echo "Wrote $(wc -l < "${DATA_DIR}/first_14k_snps.txt") SNP IDs"

echo ""
echo "=========================================="
echo "Step 3: GWAS on training set (PLINK2)"
echo "=========================================="
# Linear regression on 0/1 phenotype, training individuals only.
# --1 tells PLINK2 the phenotype is coded 0=control / 1=case.
plink2 \
  --bfile "${PLINK_PREFIX}" \
  --keep "${DATA_DIR}/train.keep" \
  --extract "${DATA_DIR}/first_14k_snps.txt" \
  --1 \
  --pheno "${DATA_DIR}/simulated_cpbayes_01.pheno" \
  --pheno-name PHENO \
  --glm allow-no-covars \
  --out "${DATA_DIR}/gwas_cpbayes_train"

echo ""
echo "GWAS output files:"
ls -la "${DATA_DIR}/gwas_cpbayes_train".*

echo ""
echo "=========================================="
echo "Step 4: LDpred2-auto"
echo "=========================================="
Rscript ldpred2.R

echo ""
echo "=========================================="
echo "Step 5: Fine-tune SNPbag"
echo "=========================================="
cd SNPbag

# Build the PLINK fileset that has the simulated phenotype in the .fam.
# We copy bed/bim and use the new fam.
cp "TestData/NewSyn_100k.bed" "TestData/NewSyn_100k_cpbayes.bed"
cp "TestData/NewSyn_100k.bim" "TestData/NewSyn_100k_cpbayes.bim"
# The cpbayes fam was already written by simulate_pheno_cpbayes.R
ln -sf "NewSyn_100k_cpbayes.fam" "TestData/NewSyn_100k_cpbayes.fam"

python3 finetune_phenotype.py \
  --pretrained-checkpoint ../"${PRETRAINED_CKPT}" \
  --plink-prefix TestData/NewSyn_100k_cpbayes \
  --phenotype-kind linear \
  --max-snps 14000 \
  --epochs 20 \
  --batch-size 24 \
  --encoder-lr 1e-5 \
  --head-lr 1e-4 \
  --val-fraction 0.10 \
  --test-fraction 0.20 \
  --seed 42 \
  --num-workers 8 \
  --save-dir checkpoints/finetune_cpbayes \
  --analysis-dir checkpoints/analysis_cpbayes

cd ..

echo ""
echo "=========================================="
echo "Step 6: Plot ROC curves and comparison"
echo "=========================================="
Rscript PlotComparison.R

echo ""
echo "Pipeline complete."
