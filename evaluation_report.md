# Evaluation Report — Astronomical Image Classifier

## Overview

This report covers the evaluation of three trained models:
- **Main Classifier** : 7-class astronomical object classifier
- **Planetary Sub-classifier** : 10-class planet and moon identifier
- **Nebula Sub-classifier** : 5-class nebula type identifier

All models use EfficientNet-B0 pretrained on ImageNet, fine-tuned using a 2-phase training strategy on combined astronomical datasets.

---

## Dataset Summary

| Source | Classes | Images |
|---|---|---|
| Galaxy10 DECaLS | Merging, Elliptical, Spiral, Edge-on | 16,655 |
| Kaggle space-images | Nebula, Star Cluster | 350 → augmented to 800 each |
| Planets and Moons Dataset | 10 planet/moon types | 149 each → augmented to 500 |
| Nebula Images Dataset | 5 nebula types | 57–741 → augmented to 300 each |

**Total combined dataset (main classifier): 19,055 images across 7 classes**

Train / Val / Test split: **70% / 15% / 15%** (stratified)

---

## Main Classifier — 7 Classes

### Overall Results

| Metric | Value |
|---|---|
| Test Accuracy | **93.25%** |
| Test Loss | 0.1855 |
| Macro F1 | 0.94 |
| Weighted F1 | 0.93 |

### Per-class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Merging Galaxy | 0.81 | 0.96 | 0.88 | 278 |
| Elliptical Galaxy | 0.93 | 0.93 | 0.93 | 751 |
| Spiral Galaxy | 0.96 | 0.90 | 0.93 | 975 |
| Edge-on Galaxy | 0.96 | 0.98 | 0.97 | 495 |
| Nebula | 0.95 | 0.95 | 0.95 | 120 |
| Planetary Object | 1.00 | 0.98 | 0.99 | 120 |
| Star Cluster | 0.93 | 0.95 | 0.94 | 120 |

### Confusion Matrix

```
                  Merging  Elliptical  Spiral  Edge-on  Nebula  Planetary  Star Cluster
Merging              266           5       6        1       0          0             0
Elliptical            17         697      30        7       0          0             0
Spiral                41          47     874       13       0          0             0
Edge-on                4           4       4      483       0          0             0
Nebula                 0           0       0        0     114          0             6
Planetary              0           0       0        0       0        118             2
Star Cluster           0           0       0        0       6          0           114
```

### Key Observations

- **Edge-on Galaxy** achieves the highest F1 (0.97) — the elongated disk morphology is visually distinct
- **Merging Galaxy** has the lowest recall (0.96 precision, 0.81 precision reversed) due to visual similarity with Spiral galaxies at low resolution
- **Planetary Object, Nebula, Star Cluster** all perform strongly despite being sourced from a different dataset and augmented from small original counts
- Galaxy classes show zero confusion with Nebula/Planetary/Star Cluster classes — the domain boundary is cleanly learned

---

## Planetary Sub-classifier — 10 Classes

### Overall Results

| Metric | Value |
|---|---|
| Test Accuracy | **100.00%** |
| Test Loss | ~0.00 |

### Per-class Performance

All 10 classes achieved perfect precision, recall, and F1-score of **1.00**.

| Class | F1-score |
|---|---|
| Mercury | 1.00 |
| Venus | 1.00 |
| Earth | 1.00 |
| Mars | 1.00 |
| Jupiter | 1.00 |
| Saturn | 1.00 |
| Uranus | 1.00 |
| Neptune | 1.00 |
| Pluto | 1.00 |
| Moon | 1.00 |

### Key Observations

- The model reached 100% validation accuracy within the first epoch of Phase 1 training
- Planets and moons are visually highly distinct — rings (Saturn), banding (Jupiter), colour (Mars, Neptune), cratering (Moon, Pluto)
- Zero confusion across all 750 test images

---

## Nebula Sub-classifier — 5 Classes

### Overall Results

| Metric | Value |
|---|---|
| Test Accuracy | **89.87%** |
| Test Loss | 0.3438 |
| Macro F1 | 0.90 |
| Weighted F1 | 0.90 |

### Per-class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Dark Nebula | 0.80 | 0.89 | 0.84 | 45 |
| Emission Nebula | 0.90 | 0.91 | 0.91 | 121 |
| Planetary Nebula | 0.90 | 0.94 | 0.92 | 50 |
| Reflection Nebula | 0.93 | 0.87 | 0.90 | 45 |
| Supernova Remnants | 0.97 | 0.87 | 0.92 | 45 |

### Confusion Matrix

```
                    Dark  Emission  Planetary  Reflection  Supernova
Dark Nebula           40         3          0           2          0
Emission Nebula        6       110          4           0          1
Planetary Nebula       1         1         47           1          0
Reflection Nebula      3         3          0          39          0
Supernova Remnants     0         5          1           0         39
```

### Key Observations

- **Dark Nebula** is the weakest class (0.84 F1) — dark nebulae are dense dust clouds that appear as dark patches against star fields, making them harder to distinguish from backgrounds
- **Supernova Remnants** achieve the highest precision (0.97) — their chaotic filamentary structure is visually distinctive
- **Planetary Nebula** performs strongly (0.92 F1) — the characteristic shell/ring structure is highly recognisable

---

## Real-World Inference

The full pipeline was tested on three real-world images not seen during training:

| Image | Expected | Predicted | Confidence |
|---|---|---|---|
| Pillars of Creation | Nebula / Emission Nebula | Nebula → Emission Nebula | 99.8% / 95%+ |
| Face-on Spiral Galaxy | Spiral Galaxy | Spiral Galaxy | 98.3% |
| Saturn | Planetary / Saturn | Planetary → Saturn | 100% / 96.2% |

All three correctly classified with high confidence.

---

## Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) was implemented from scratch on the final convolutional block of EfficientNet-B0. Results confirmed the model is focusing on scientifically meaningful regions:

- **Pillars of Creation** — attention on the dense gas pillar structures
- **Spiral Galaxy** — attention on the galactic core and disk
- **Saturn** — attention precisely traces the planetary body and ring system

---

## Training Configuration

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| Backbone | Frozen | Unfrozen |
| Optimizer | AdamW | AdamW |
| Learning Rate | 1e-3 | 1e-4 |
| Epochs | 10 | 15 |
| Batch Size | 32 | 32 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Sampler | WeightedRandomSampler | WeightedRandomSampler |

---

## Model Summary

| Model | Parameters | Classes | Test Accuracy |
|---|---|---|---|
| Main Classifier | 4,337,283 | 7 | 93.25% |
| Planetary Sub-classifier | 4,337,540 | 10 | 100.00% |
| Nebula Sub-classifier | 4,337,027 | 5 | 89.87% |