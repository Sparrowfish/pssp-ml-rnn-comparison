# Protein Secondary Structure Prediction

A sequential machine learning study predicting three-state protein secondary structure (α-helix, β-strand, coil) from amino acid sequence. Each step is motivated by a concrete diagnosis of the previous model's failure mode: from a leaked, inflated baseline through encoding fixes, architecture changes, and pretrained representations.

---

## Task Definition

Given a protein primary sequence, assign a secondary structure label to every residue:

| Label | Structure | Description |
|---|---|---|
| **H** | α-helix | Local hydrogen bonding every 4 residues |
| **E** | β-strand | Extended conformation; H-bonds to a distant partner strand |
| **C** | Coil | Loops, turns, disordered regions |

Labels come from DSSP, reduced from 8 states to 3: H←{H,G,I}, E←{E,B}, C←{T,S,C,-}.

---

## Development Narrative

### 1. Inflated baseline (89%) → corrected protocol (60%)

The first dataset queried PDB without diversity constraints and split at the **residue level**. This leaked context (windows from the same protein appear in both train/test) and allowed near-duplicate proteins (myoglobin/lysozyme variants) across the split, producing a meaningless 89% accuracy.

Fixed by: splitting at the **protein level**, sampling one representative per 30%-identity cluster (the CB513/CASP-style protocol), and keeping one chain per PDB entry. Accuracy fell to **60%** which was the real starting point.

### 2. Traditional ML on local windows (60% → 69%)

| Change | Result |
|---|---|
| Window size 5→15 | 60%→66% (21 gave no further gain |
| Class weighting | +3pp E-recall |
| XGBoost over Random Forest | 66% accuracy, E-recall 0.42 |
| Global composition features | **Counterproductive** - 62% accuracy, E-recall 0.27 (protein-level signal is noisy per-residue) |
| BLOSUM62 over one-hot | 69% accuracy, E-recall 0.50 (largest single gain) |

**Structural ceiling**: no window size fixes β-strand recall, because strands pair with partners that can be hundreds of residues away — a non-locality no fixed window can resolve. This diagnosis motivated every step after it.

### 3. Sequence models — solving non-locality (70%+)

- **Notebook 4 - BiLSTM (BLOSUM62)**: processes the whole sequence in both directions, so every residue's representation includes full long-range context. **70% accuracy, E-recall 0.57.**
- **Notebook 5 - Transformer (BLOSUM62)**: self-attention connects any two residues in one hop, instead of the BiLSTM's step-by-step chain. This is the more architecturally direct fix for strand pairing, at the cost of needing more data than a ~480-protein training set ideally provides. 
- **Notebook 6 - BiLSTM (ESM-2 embeddings)**: same BiLSTM as notebook 4, but BLOSUM62 (a fixed, generic 20-dim table) is replaced with per-residue embeddings from ESM-2, a protein language model pretrained on hundreds of millions of sequences. Isolates the effect of encoding quality alone.
- **Notebook 7 - CBRNN (CNN → BiLSTM, ESM-2 embeddings)**: adds 1D convolutions ahead of the BiLSTM to explicitly detect local motifs (e.g. helix periodicity) before the BiLSTM adds long-range context (the architecture used by current state-of-the-art PSSP systems such as Porter6 and SPOT-1D-LM).

---

## Summary of Results

| Model | Encoding | Split | Accuracy | E-Recall | Notes |
|---|---|---|---|---|---|
| Random Forest | One-hot | Residue-level | 89% | 0.76 | **Invalid: leakage + homology contamination** |
| Random Forest | One-hot | Protein-level | 60% | 0.28 | Corrected baseline |
| XGBoost | One-hot | Protein-level | 66% | 0.42 | |
| XGBoost | One-hot + composition | Protein-level | 62% | 0.27 | Composition features harmful |
| XGBoost | BLOSUM62 | Protein-level | 69% | 0.50 | Best traditional ML result |
| BiLSTM | BLOSUM62 | Protein-level | 70% | 0.57 | Best result with a static encoding |
| Transformer | BLOSUM62 | Protein-level | 54% | 0.29 | Notebook 5 |
| BiLSTM | ESM-2 | Protein-level | 80% | 0.76 | Notebook 6 |
| CBRNN | ESM-2 | Protein-level | 81% | 0.72 | Notebook 7 |


---

## Limitations and Potential Extensions

- **Small dataset**: ~500–600 proteins is small by deep learning standards, and is the likely reason a from-scratch Transformer did not outperform the BiLSTM. Attention lacks recurrence's built-in sequential bias and needs more data to compensate.
- **CNN padding leakage (notebook 7)**: `nn.Conv1d` has no notion of padded vs. real positions, so it convolves slightly into zero-padding near the end of shorter sequences in a batch.
- **PSSM as a complementary feature**: literature suggests real per-protein evolutionary profiles (PSSM) still add a small amount of signal even on top of PLM embeddings. This is worth testing as a concatenated feature with ESM-2.
- **Fine-tuning ESM-2**: notebooks 6–7 use frozen embeddings; unfreezing the last few ESM-2 layers (at a much lower learning rate than the head) is the natural next step between "frozen features" and full end-to-end training.
- **Per-protein error analysis**: characterise which structural classes (all-β proteins, membrane proteins, intrinsically disordered regions) drive the residual error.
