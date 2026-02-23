# Datium Data Science Assessment – Section B
## Visual Analysis of Vehicle Images

This document is a **design and strategy report** — not a training run (no labelled image dataset is provided).
The goal is to demonstrate how a practical end-to-end image-intelligence system would be designed for an automotive assessment business.

---

## 1. Problem Framing

### Business Context
Datium receives **user-submitted vehicle images** as part of commercial assessments (e.g. trade-in, insurance, fleet valuation).
The images are uncontrolled: variable lighting, angles, backgrounds, and image quality.

### Possible Actionable Outputs

| Signal | Value to Business |
|--------|------------------|
| **Damage detection** | Identify dents, scratches, cracks → adjust valuation downward |
| **Vehicle condition score** | Aggregate visual health score (0–10) |
| **Colour identification** | Automate a common data entry field |
| **Body style classification** | Sedan vs SUV vs Ute → supports market segmentation |
| **Odometer/dash reading** | OCR from interior photo → cross-validate KM field |
| **Image quality gating** | Reject blurry/incomplete images before downstream processing |

### Chosen Focus: Damage Detection + Condition Scoring
This is the highest-value signal: damage directly affects resale price, it is labour-intensive to assess manually, and modern vision models are well-suited to it.

---

## 2. Data Assumptions & Challenges

### What we assume about incoming images
- JPEG/PNG, typical smartphone quality (2–12 MP)
- 1–10 photos per vehicle (exterior: front, rear, sides; interior; odometer)
- Submitted by non-expert users — inconsistent framing, glare, obstructions

### Key data challenges

| Challenge | Mitigation |
|-----------|------------|
| **No labelled damage dataset** | Combine public datasets (CARDD, VCoR) with in-house labelling (Label Studio) |
| **Class imbalance** (most cars undamaged) | Focal loss, oversampling damaged examples, threshold tuning |
| **Variable image angle** | Multi-view aggregation; angle classification as pre-step |
| **Low-quality images** | Image quality classifier as first gate; feedback to user |
| **Privacy (licence plates, faces)** | Auto-blur in pre-processing pipeline |
| **Adversarial submissions** (hiding damage) | Anomaly detection; consistency checks against odometer KM |

---

## 3. Modelling Approaches

### 3.1 Approach A – Fine-tuned CNN Classifier (Baseline)

```
Input image → EfficientNet-B3 backbone (ImageNet pretrained)
           → Global average pooling
           → Dense head → Condition score (regression) + Damage present (binary)
```

- **Pros:** Fast inference, well-understood, easy to deploy
- **Cons:** No spatial localisation — can't say *where* the damage is

---

### 3.2 Approach B – Object Detection for Damage Localisation

```
Input image → YOLOv11 / RT-DETRv2
           → Bounding boxes: {scratch, dent, crack, rust, broken_glass}
           → Damage area fraction → condition score formula
```

- **Pros:** Interpretable (show bounding boxes in app), actionable per-damage-type pricing
- **Cons:** Requires polygon/bbox annotations (expensive)

---

### 3.3 Approach C – Vision–Language Foundation Model (Zero/Few-Shot)

```
Input image + prompt → GPT-5.2 / Gemini 3.1 Pro
                     → Structured JSON: {has_damage, severity, affected_panels, notes}
```

Prompt example:
```
"Inspect this vehicle image. Return JSON:
  has_damage (bool), severity (none/minor/moderate/severe),
  affected_panels (list), confidence (0–1).
  Focus only on visible damage to paintwork, glass, and body panels."
```

- **Pros:** No training data needed to start; handles edge cases well; human-readable explanations
- **Cons:** API cost at scale; latency; not suitable for offline/edge deployment

---

### 3.4 Recommended Hybrid Architecture

```mermaid
flowchart TD
    A[Image Ingestion API] --> B["Image Quality Gate\nblur · coverage · brightness"]
    B -->|PASS| C["Angle Classifier\nfront · rear · side · interior"]
    B -->|FAIL| F[Feedback to User]
    C --> D["Damage Detection\nYOLOv11"]
    C --> E["Colour ID\nK-means / ViT embed."]
    C --> G["Odometer OCR\nTrOCR"]
    D --> H["Score Aggregation & Rules\ncondition_score = f(damage, age, km_visual, panel_count)"]
    E --> H
    G --> H
    H --> I["Valuation Adjustment Signal\n→ feeds into Section A model"]
```

---

## 4. Evaluation Strategy

### Damage Detection (object detection)
- **mAP@0.5** per damage class
- **Precision / Recall** at business-relevant operating points
  (prefer high recall for severe damage — false negatives cost more than false positives)

### Condition Score (regression)
- **MAE / RMSE** vs human assessor ground truth
- **Pearson / Spearman correlation** with final adjusted sale price
- **Inter-rater agreement** (Cohen's κ on ordinal score buckets)

### End-to-End Business Metric
- **Δ valuation accuracy**: Does adding the image score reduce MAE in Section A's price model?
  Measure: `MAE_with_image_feature` vs `MAE_without`

---

## 5. Practical Constraints & Trade-offs

| Constraint | Consideration |
|------------|---------------|
| **Data labelling cost** | Start with foundation model (Approach C) to bootstrap labels cheaply; then fine-tune YOLO |
| **Inference latency** | YOLOv11-nano gives <50 ms on CPU; adequate for near-real-time assessment |
| **Model updates** | Damage styles evolve (EV battery enclosures, new materials); plan quarterly re-training |
| **Explainability** | Bounding boxes + severity labels are explainable to assessors and customers |
| **Reliability** | Always provide a human override path; model is advisory, not authoritative |
| **Scale** | At 10k assessments/day with 5 images each → 50k images/day; batch GPU inference is cost-effective |

---

## 6. MVP Implementation Roadmap

### Phase 1 – Foundation (2–4 weeks)
- Image quality gate using OpenCV (Laplacian variance for blur, coverage heuristics)
- Zero-shot damage classification with a vision LLM → generate weak labels
- Evaluate LLM output quality against 200 human-labelled images

### Phase 2 – Supervised Model (4–8 weeks)
- Label 2,000–5,000 images (bounding boxes) using Label Studio
- Fine-tune YOLOv11-m on damage classes
- A/B test: does image-derived condition score improve Section A model's MAE?

### Phase 3 – Production Integration (ongoing)
- Wrap in FastAPI microservice; integrate with vehicle assessment workflow
- Active learning loop: flag uncertain predictions for human review → grow labelled set
- Drift monitoring: track prediction distribution shifts over time
