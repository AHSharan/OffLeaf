# OffLeaf — Results

Headline numbers, kept in git because `runs/` is not tracked. Raw JSON in
`results/`. Update as runs land.

**Status:** E0 baseline complete (10 epochs, 92.8 min, best val 0.99797 @ epoch
9). All numbers are from the **final** checkpoint unless a row says otherwise.
E1a seeds 1–2 and E1c are running; E1b, E1d and E2 are pending.

---

## The five headline numbers

| # | finding | number |
|---|---|---|
| 1 | Background alone classifies PlantVillage | **43.9%** vs 2.6% chance (**16.7×**) |
| 2 | Lab → field collapse | **99.78% → 19.53%** (**80 points**) |
| 3 | Attribution mass off the leaf, on field images | **49.2%** |
| 4 | Background share of the lab-to-field gap | **62.6%** |
| 5 | Training to convergence costs field accuracy | **−6.0 points** for +0.24 lab |

---

## 1. Background alone classifies PlantVillage

Replication of Noyan (2022) on our own splits. Random forest on 8 background
corner pixels (24 RGB features). No leaf pixels, no neural network, no GPU.

| | |
|---|---|
| accuracy | **0.4390** [0.4262, 0.4522] |
| chance (38 classes) | 0.0263 |
| ratio | **16.7× chance** |
| train / test images | 12,000 (of 43,444) / 5,431 |

Noyan reported 0.490; ours is slightly lower on a subsample with a different
classifier. **This is a replication, not a contribution — cite Noyan.** Its job
is to establish that the shortcut exists in *our* splits, so every later number
is anchored to it.

`results/noyan_test_seed0.json`

---

## 2. The lab-to-field collapse

E0 ResNet-50, trained on PlantVillage only, evaluated zero-shot on PlantDoc.

| | |
|---|---|
| PlantVillage test (lab) | **0.9978** [0.9965, 0.9991], n=5431 |
| lab macro F1 | 0.9968 |
| PlantDoc (field) | **0.1953** [0.1802, 0.2101], n=2580 |
| field macro F1 | 0.1380 |

**An 80-point collapse.**

All of PlantDoc is held out — the model never trains on it — so both its train
and test splits are legitimate evaluation data, which tightens the CIs.

**Comparability:** Frontiers in Plant Science (2026) reports 99.73 → 32.05 on
this dataset pair using **21 shared crop-disease pairs**. We evaluate **all 28
PlantDoc classes**, a harder setting. State this when citing, so the difference
reads as a methodological choice rather than a discrepancy.

`results/E0_plantdoc_final.json`, `results/E0_labtest_final.json`

---

## 3. Where the model looks, on field images

Grad-CAM relevance mass against SAM leaf masks.

| | |
|---|---|
| **`offleaf_mass`** | **0.4920** [0.4767, 0.5047] |

Nearly half of all attribution mass falls **off the leaf**.

This matters methodologically. It is the *attention-based* measure, and it agrees
with the *behavioural* measure in section 4, which uses no heatmap at all. Two
independent instruments pointing the same way is what licenses using attribution
maps as evidence in this domain. Had they disagreed, that would itself have been
the finding — and it would have undermined the heatmap-based claims common in
this literature.

**Caveat:** leaf masks are SAM-generated, not human. The lesion/pseudo rule does
not cover them because leaf-vs-background is far less ambiguous than a lesion
boundary — but state the quality: **4% HSV fallback on PlantVillage, 18% flagged
unreliable on PlantDoc.**

---

## 4. Counterfactual background swap — the main result

200 tomato leaves per cell, 1,000 composites. Same leaf, different background.

| cell | accuracy | what it isolates |
|---|---|---|
| `paste_control` | **0.910** | compositing artefacts only — leaf on its *own* background |
| `lab_plain` | **0.865** | + a *different* lab background |
| `lab_field` | **0.405** | + a **field** background (leaf pixel-identical) |
| `field_plain` | **0.170** | field leaf, lab background |
| `field_field` | **0.130** | unmodified field photo |

### Background share of the lab-to-field gap: **62.6%**

### The asymmetry — lead with this

- **Lab leaves:** swapping grey for foliage costs **46 points** (0.865 → 0.405),
  with the leaf pixel-identical.
- **Field leaves:** the reverse swap buys **4 points** (0.130 → 0.170).

Background dependence is **specific to the training regime**. Once the leaf
itself looks unfamiliar, the background is no longer what is failing.

This sets an honest ceiling on the interventions: they should make a lab-trained
model resist background change; they will **not** fix leaf-appearance shift. Say
so before a reviewer does.

### Caveats

- **Compositing costs ~9 points** (0.998 lab test → 0.910 paste_control). Every
  cell carries that cost, which is why cells are compared **to each other**,
  never to raw lab accuracy.
- `paste_control` scores above `lab_plain` — a leaf's own background is easier
  than another lab image's. Informative, not a bug.
- Background plates use `cv2.inpaint` and carry **visible ghosts** of the removed
  leaf. Usable but not clean; better inpainting would strengthen the benchmark.
- 200 leaves per cell is modest. Bootstrap CIs are in the JSON.

`results/E0_counterfactual_final.json`

---

## 5. Training to convergence *hurts* field transfer

The same E0 run at two checkpoints:

| | epoch 3 | epoch 9 (final) | change |
|---|---|---|---|
| Lab val accuracy | 0.9956 | **0.9980** | +0.24 pp |
| **Field accuracy** | **0.2550** [0.2395, 0.2721] | **0.1953** [0.1802, 0.2101] | **−5.97 pp** |
| `offleaf_mass` | 0.4886 | 0.4920 | +0.34 pp |
| Background share of gap | 70.3% | 62.6% | −7.7 pp |

The field confidence intervals **do not overlap**. Six more epochs bought 0.24
points of lab accuracy and cost 6 points of field accuracy.

**Implication:** selecting a checkpoint by validation accuracy — standard
practice, and what `train.py` does — actively selects the *worse* field model.
Longer training fits the shortcut harder.

**Caveat, state plainly:** **two checkpoints, not a curve.** Only the best-val
checkpoint is kept, so intermediate epochs no longer exist. We can say epoch 3
beat epoch 9; we cannot yet say the decline is monotonic. Claiming the trend
needs a re-run that saves every epoch (~93 min training + minutes of eval).

`results/E0_plantdoc_interim.json` vs `results/E0_plantdoc_final.json`

---

## 6. Background removal costs nothing in the lab

| run | best val accuracy |
|---|---|
| E0 baseline | 0.99797 |
| E1c bgremoval | 0.99558 |

Deleting the background entirely barely moves lab accuracy. The model **can**
classify from leaf pixels alone — it simply does not bother to when the
background is available.

This closes an obvious objection: the background carries no genuine diagnostic
signal. The open question is only whether a model *forced* onto the leaf
transfers better.

### A deployment gotcha worth one line

Scoring the E1c checkpoint on **unmasked** field scenes gives 0.2083 — worse than
the 0.2550 baseline at the matching checkpoint. That is not background removal
failing; it is a model trained on leaves-against-blank being shown full scenes,
which is maximally out of distribution for it.

The real lesson: **background removal is a two-part commitment, training *and*
inference.** Deploy it without a segmenter in front and you are worse off than
having done nothing. `evaluate.py` now auto-applies the leaf mask for
`bgremoval` checkpoints so the mistake cannot recur.

---

## Still to come

- **E1a seeds 1–2** (running) — seed spread for the baseline
- **E1c field accuracy** with the leaf mask correctly applied
- **E1b** copy-paste, **E1d** cam_penalty at leaf level (blocked on the remaining
  ~11k PlantVillage leaf masks, generating now)
- **E2** the λ dose–response on tomato — the strongest single result
- **Human verification of 80 pseudo-masks**, which is what makes `offlesion_mass`
  metric-grade. Until then only `offleaf_mass` (leaf-level) is reportable.
