# Protein Secondary Structure Prediction

A sequential machine learning study predicting three-state protein secondary structure (α-helix, β-strand, coil) from amino acid sequence. The project progresses deliberately from a flawed baseline through a series of methodological corrections and architectural improvements, with each step motivated by a concrete diagnosis of the previous model's failure mode.

---

## Task Definition

Given a protein primary sequence, the goal is to assign a secondary structure label to every residue:

| Label | Structure | Description |
|---|---|---|
| **H** | α-helix | Local hydrogen bonding every 4 residues along the chain |
| **E** | β-strand | Extended conformation; hydrogen bonds to a distant partner strand |
| **C** | Coil | All other conformations (loops, turns, disordered regions) |

Labels are derived from DSSP (Define Secondary Structure of Proteins), which assigns structure from atomic coordinates. The 8-state DSSP output (H, G, I, E, B, T, S, C) is reduced to 3 states: H←{H,G,I}, E←{E,B}, C←{T,S,C,-}.

---

## Repository Structure

```
.
├── 01_data_collection.ipynb       # PDB query, DSSP labelling, dataset construction
├── 02_baseline_models.ipynb       # Random Forest and XGBoost with one-hot encoding
├── 03_blosum62_encoding.ipynb     # BLOSUM62 substitution matrix encoding
├── 04_bilstm_blosum62.ipynb       # Bidirectional LSTM with BLOSUM62 input
└── README.md
```

---

## Development Narrative

### 1. Initial Dataset and Baseline (Inflated: 89%)

The first dataset was constructed by querying the RCSB PDB for X-ray crystallography entries and downloading the first N results. Secondary structure was assigned with DSSP and residue-level windows of size 11 were extracted as features using one-hot encoding. A Random Forest classifier was trained on a random 80/20 residue-level split.

```
              precision    recall  f1-score   support
           H       0.87      0.89      0.88      3158
           E       0.92      0.76      0.83      1401
           C       0.90      0.94      0.92      4433
    accuracy                           0.89      8992
```

The 89% figure is an artifact of two compounding problems:

**Data leakage via residue-level splitting.** Splitting at the residue level rather than the protein level means that for a protein of length L, approximately 80% of its residues appear in training and 20% in test. Because window features are computed from overlapping local neighbourhoods, the training set contains near-complete context for every test residue. The model is not generalising; it is interpolating within proteins it has already seen.

**Homology contamination.** Querying PDB without diversity constraints retrieves heavily over-represented proteins; myoglobin and lysozyme variants account for a disproportionate share of early PDB entries. Near-identical sequences appear in both train and test sets, further inflating apparent performance.

---

### 2. Correcting the Evaluation Protocol

**Residue-level → protein-level split.** The split was moved to the protein level: all residues from a given protein appear exclusively in train or test, never both.

**Sequence identity clustering.** PDB's pre-computed 30% sequence identity clusters were used to sample the dataset. One representative per cluster was selected, and the train/test split was performed at the cluster level to prevent homologous proteins from spanning both sets. This is the standard protocol in structural bioinformatics benchmarks (e.g. CB513, CASP evaluations).

**One chain per PDB entry.** Multi-chain assemblies (ribosomes, viral capsids) can contribute dozens of chains from a single structure, skewing the dataset toward those protein families. Only the longest valid chain per PDB entry was retained.

After applying these corrections, the dataset comprised ~500 structurally and sequentially diverse proteins. Accuracy under the corrected protocol fell to **60%** — reflecting genuine generalisation performance.

---

### 3. Systematic Feature and Model Improvements

#### 3.1 Window Size (5 → 15: +6% Accuracy | 15 → 21: No Improvement)

Increasing the window from 5 to 15 residues improved accuracy from 60% to 66%, confirming that secondary structure depends on local sequence context beyond the immediate residue. Expanding further to 21 produced no gain. α-helix propensity is largely determined by local sequence patterns capturable within a window of ~15 residues. β-strand identity, however, is defined by hydrogen bonding between strands that may be hundreds of residues apart in sequence — a fundamental non-locality that no window size can resolve. Beyond ~15 residues, the window ceiling is a structural limitation of the approach, not a tunable hyperparameter.

#### 3.2 Class Weighting: Modest E Improvement

Setting `class_weight="balanced"` in the Random Forest increased β-strand recall by approximately 3 percentage points by penalising misclassification of the minority strand class more heavily. The effect was limited because the underlying feature representation still lacked strand-specific signal.

#### 3.3 XGBoost: Accuracy 66%, E-Recall 0.42

Replacing the Random Forest with gradient-boosted trees (XGBoost, 200 estimators, max depth 6) improved both overall accuracy and strand recall.

```
              precision    recall  f1-score   support
           C       0.65      0.70      0.67     12974
           E       0.60      0.42      0.49      6537
           H       0.68      0.73      0.70     13698
    accuracy                           0.66     33209
```

XGBoost's sequential residual fitting captures interaction terms between window positions more effectively than the independent tree averaging of Random Forest.

#### 3.4 Global Composition Features: Counterproductive

Appending the global amino acid composition of each protein (a 20-dimensional frequency vector) to every residue's window features reduced accuracy to 62% and E-recall from 0.42 to 0.27. The degradation is explained by the high within-class variance of protein composition: a protein's overall amino acid frequencies provide a noisy, protein-level signal that is inconsistent across residues of that protein. In a flat feature vector, these 20 values compete with the 300 window features during tree splitting without offering discriminative per-residue information. The result is that composition features introduce noise rather than signal at the residue level, and the model's ability to identify strands deteriorates substantially.

#### 3.5 BLOSUM62 Encoding: Accuracy 69%, E-Recall 0.50

One-hot encoding is biochemically uninformative; it treats all amino acid substitutions as equally large. BLOSUM62 replaces the binary indicator vector with a 20-dimensional row from the BLOSUM62 substitution matrix, where each element encodes the log-odds probability of a substitution observed in aligned blocks of evolutionarily related sequences. Biochemically similar amino acids (e.g. Leu/Ile, Asp/Glu) have similar BLOSUM62 vectors, enabling the model to generalise across conservative substitutions.

```
              precision    recall  f1-score   support
           C       0.66      0.72      0.69     12974
           E       0.64      0.50      0.56      6537
           H       0.73      0.75      0.74     13698
    accuracy                           0.69     33209
```

E-recall improved from 0.42 to 0.50 — the largest single gain from any feature engineering step.

---

### 4. Bidirectional LSTM: Accuracy 70%, E-Recall 0.57

Analysis of the XGBoost confusion matrix revealed that β-strand errors were split approximately equally between H and C predictions, indicating the model had no reliable signal for strand identification which means near-random classification for that class. This is a structural problem: β-strands are defined by inter-strand hydrogen bonds between sequence positions that may be separated by hundreds of residues, and no local window representation can capture this non-local dependency.

A Bidirectional LSTM processes the entire protein sequence in both directions before making any prediction. Every residue's output representation is conditioned on the full preceding (forward) and following (backward) sequence context, directly addressing the non-locality limitation.

**Architecture:**
- Input: BLOSUM62 vectors, 20-dimensional per residue
- BiLSTM: 2 layers, 128 hidden units per direction (256 total), dropout 0.3
- Classifier head: Linear(256 → 3)
- Loss: CrossEntropyLoss with `ignore_index=-1` for padded positions
- Optimiser: Adam, lr=1e-3, gradient clipping at norm 1.0

```
              precision    recall  f1-score   support
           C       0.65      0.72      0.69     12974
           E       0.64      0.57      0.60      6537
           H       0.78      0.74      0.76     13698
    accuracy                           0.70     33209
```

β-strand recall reached 0.57, compared to 0.34 at the corrected baseline. The improvement confirms that global sequence context carries information about strand propensity that local windows cannot access.

---

## Summary of Results

| Model | Encoding | Split | Accuracy | E-Recall | Notes |
|---|---|---|---|---|---|
| Random Forest | One-hot | Residue-level | 89% | 0.76 | **Invalid — data leakage + homology contamination** |
| Random Forest | One-hot | Protein-level | 60% | 0.28 | Corrected baseline |
| Random Forest | One-hot | Protein-level | 65% | 0.31 | Window=21, no improvement over window=11 |
| XGBoost | One-hot | Protein-level | 66% | 0.42 | Gradient boosting outperforms RF |
| XGBoost | One-hot + Composition | Protein-level | 62% | 0.27 | Global composition features harmful at residue level |
| XGBoost | BLOSUM62 | Protein-level | 69% | 0.50 | Best traditional ML result |
| **BiLSTM** | **BLOSUM62** | **Protein-level** | **70%** | **0.57** | **Best overall** |

---

## Requirements

```
python >= 3.10
biopython
torch >= 2.0
scikit-learn
xgboost
pandas
numpy
matplotlib
mkdssp
```

```bash
pip install biopython torch scikit-learn xgboost pandas numpy matplotlib
```

DSSP binary (`mkdssp`) must be installed separately:

```bash
# Ubuntu/Debian
sudo apt install dssp

# conda
conda install -c salilab dssp
```

---

## Reproducing Results

Run notebooks in order. Notebook 1 downloads PDB files and writes `protein_sequences_ss.csv`; all subsequent notebooks load this file.

```bash
jupyter notebook 01_data_collection.ipynb   # requires network access, ~1–2 hrs
jupyter notebook 02_baseline_models.ipynb
jupyter notebook 03_blosum62_encoding.ipynb
jupyter notebook 04_bilstm_blosum62.ipynb
```

`random_seed=42` is set throughout. Minor numerical differences may arise from non-deterministic CUDA operations in notebook 4.

---

## Limitations and Potential Extensions

The remaining gap to state-of-the-art (~85–90%) is primarily explained by the absence of evolutionary information. Protein language models such as ESM-2 (Lin et al., 2023) are pre-trained on 250 million sequences and produce per-residue embeddings that implicitly encode conservation, co-evolution, and structural propensity. Replacing BLOSUM62 input with ESM-2 embeddings is the highest-leverage extension of this work.

Additional directions:
- **Per-protein error analysis**: characterise which structural classes (all-β proteins, membrane proteins, intrinsically disordered regions) drive the residual error
- **Attention-based architectures**: transformer encoder with self-attention over the full sequence, enabling direct inspection of long-range dependencies
- **Larger dataset**: the current ~500-protein dataset is small by deep learning standards; scaling to several thousand proteins with the same diversity protocol would likely improve BiLSTM performance substantially
