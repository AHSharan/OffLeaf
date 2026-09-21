# OffLeaf — Results

Headline numbers, kept in git because `runs/` is not tracked. Update as runs land.

**Status:** E0 baseline complete (10 epochs, 92.8 min, best val 0.99797 @ epoch 9).
Numbers below are from the **final** checkpoint unless marked otherwise.

---

## 1. Background alone classifies PlantVillage

Replication of Noyan (2022) on our own splits. Random forest, 8 background
corner pixels (24 RGB features), no leaf pixels, no neural network.

| | |
|---|---|
| accuracy | **0.4390** [0.4262, 0.4522] |
| chance (38 classes) | 0.0263 |
| ratio | **16.7x chance** |
| train images | 12,000 (of 43,444) |
| test images | 5,431 |

Noyan reported 0.490. Ours is slightly lower on a subsample and a different
classifier. **This is a replication, not a contribution — cite Noyan.** Its job
is to establish that the shortcut exists in *our* splits.

`runs/noyan_test/seed0.json`

---

## 2. The lab-to-field collapse

E0 ResNet-50, trained on PlantVillage only, evaluated zero-shot on PlantDoc.

| | |
|---|---|
| PlantVillage val (lab) | **0.9956** |
| PlantDoc (field) | **0.2550** [0.2395, 0.2721] |
| macro F1 (field) | 0.1756 |
| images | 2,580 across 28 classes |

**74-point collapse.**

All of PlantDoc is held out — the model never trains on it — so both its train
and test splits are legitimate evaluation data.

**Comparability:** Frontiers in Plant Science (2026) reports 99.73 → 32.05 on
this dataset pair using **21 shared crop-disease pairs**. We evaluate **all 28
PlantDoc classes**, a harder setting. State this when citing, so the difference
reads as a methodological choice rather than a discrepancy.

`runs/E0_resnet50/0/eval_plantdoc.json`

---

## 3. Where the model looks, on field images

Grad-CAM relevance mass, leaf masks from SAM.

| | |
|---|---|
| **offleaf_mass** | **0.4886** [0.4744, 0.5013] |

Nearly half of all attribution mass falls **off the leaf** on field images.

This matters methodologically: it is the *attention-based* measure, and it agrees
with the *behavioural* measure in section 4, which uses no heatmap at all. Two
independent instruments pointing the same way is what licenses using attribution
maps as evidence in this domain — had they disagreed, that would itself have been
the finding.

---

## 4. Counterfactual background swap — the main result

200 tomato leaves per cell, 1,000 composites. Same leaf, different background.

| cell | accuracy | what it isolates |
|---|---|---|
| `paste_control` | **0.770** | compositing artefacts only — leaf on its *own* background |
| `lab_plain` | **0.685** | + a *different* lab background |
| `lab_field` | **0.295** | + a **field** background (same leaf pixels) |
| `field_plain` | **0.140** | field leaf, lab background |
| `field_field` | **0.130** | unmodified field photo |

### Background share of the lab-to-field gap: **70.3%**

### The asymmetry — the part worth presenting

- **Lab leaves:** swapping grey for foliage costs **39 points** (0.685 → 0.295),
  with the leaf pixel-identical.
- **Field leaves:** swapping foliage for grey buys **1 point** (0.130 → 0.140).

Background dependence is **specific to the training regime**. Once the leaf
itself looks unfamiliar, the background is no longer what is failing. This sets
an honest ceiling on the interventions: they should make a lab-trained model
resist background change; they will not fix leaf-appearance shift.

### Caveats to state

- Compositing costs **22.6 points** overall (0.9956 clean val → 0.770
  paste_control). Every cell carries that cost, which is why cells are compared
  to **each other**, never to raw val accuracy.
- `paste_control` scores *above* `lab_plain`, which is itself informative: a
  leaf's own background is easier than another lab image's.
- Background plates are built with `cv2.inpaint` and carry **visible ghosts** of
  the removed leaf. Usable, but not clean — a stronger benchmark would use
  better inpainting.
- 200 leaves per cell is modest. Bootstrap CIs are in the JSON.

`runs/E0_resnet50/0/eval_counterfactual.json`

---

## 5. Background removal costs nothing in the lab

| run | val accuracy |
|---|---|
| E0 baseline | 0.9956 |
| E1c bgremoval (epoch 2) | 0.9917 |

Deleting the background entirely barely moves lab accuracy. The model **can**
classify from leaf pixels alone — it simply does not bother to when the
background is available.

This closes an obvious objection: the background does not carry genuine
diagnostic signal. The open question is only whether a model *forced* onto the
leaf transfers better.

---

## Still to come

- E0 final (epoch 10) — re-run sections 2–4 against it
- E1a/b/c/d: the three interventions vs baseline, 38 classes
- E2: the lambda dose-response on tomato
- Human verification of 80 pseudo-masks, which is what makes `offlesion_mass`
  metric-grade. Until then only `offleaf_mass` (leaf-level, SAM masks) is
  reportable.

---

## 6. Training to convergence *hurts* field transfer

The same E0 run, evaluated at two checkpoints:

| | epoch 3 | epoch 9 (final) | change |
|---|---|---|---|
| Lab val accuracy | 0.9956 | **0.9980** | +0.24 pp |
| **Field accuracy (PlantDoc)** | **0.2550** [0.2395, 0.2721] | **0.1953** [0.1802, 0.2101] | **−5.97 pp** |
| `offleaf_mass` | 0.4886 | 0.4920 | +0.34 pp |
| Background share of gap | 70.3% | 62.6% | −7.7 pp |

The confidence intervals on field accuracy **do not overlap**. Six more epochs
bought 0.24 points of lab accuracy and cost 6 points of field accuracy.

**Implication:** selecting a checkpoint by validation accuracy — standard
practice, and what this repo's `train.py` does — actively selects the *worse*
field model. Longer training fits the shortcut harder.

**Caveat, state this plainly:** this is **two checkpoints, not a curve.** Only
the best-val checkpoint is saved, so the intermediate epochs no longer exist. We
can say epoch 3 beat epoch 9; we cannot yet say the decline is monotonic. To
claim the trend properly, re-run E0 saving every epoch and evaluate each on
PlantDoc — roughly 93 min of training plus a few minutes of evaluation.

### Final E0 counterfactual cells (epoch 9)

| cell | accuracy |
|---|---|
| `paste_control` | 0.910 |
| `lab_plain` | 0.865 |
| `lab_field` | 0.405 |
| `field_plain` | 0.170 |
| `field_field` | 0.130 |

Swapping only the background costs a **lab** leaf **46 points** (0.865 → 0.405).
The reverse swap buys a **field** leaf **4 points** (0.130 → 0.170). The
asymmetry from section 4 holds, and is larger on the better-trained model.

### Final E0 lab/field summary

| | |
|---|---|
| Lab test accuracy | **0.9978** [0.9965, 0.9991], n=5431 |
| Lab macro F1 | 0.9968 |
| Field accuracy | **0.1953** [0.1802, 0.2101], n=2580 |
| Field macro F1 | 0.1380 |
| `offleaf_mass` (field) | **0.4920** [0.4767, 0.5047] |

**An 80-point lab-to-field collapse.**
