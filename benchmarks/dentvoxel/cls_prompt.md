# DentVoxel Axial-Slice Classification Prompt (`dentvoxel_cls` · `cls-lenet2d-v1`)

You are participating in a blind **Weight-Space Programming (WSP)** experiment mirroring Lindholz et al. (*Weight-Space Programming for Medical Imaging*). Do **not** use web search, code execution, or external tools. You will see no example slices.

## Task
Directly write all **261 floating-point parameters** for `cls-lenet2d-v1` to classify whether a single axial dental CBCT slice intersects the **bilateral maxillary sinuses** (`label = 1`, maxillary sinus present) versus slices outside the maxillary sinuses (`label = 0`, e.g., mandibular body/roots, occlusal arch, or inferior neck).

## Input & Preprocessing
- Single axial `(Y, X)` slice extracted from a RAS+ canonical large-FOV dental CBCT scan, normalized with dental HU window `[-1000, +3000]` → `[0.0, 1.0]`, and aspect-preserving centered on a `64 × 64` canvas.
- Radiographic physics cue: Maxillary sinus slices display a pair of **large, symmetric, dark air-filled cavities (`~0.0`)** in the anterior-lateral midface bounded by thin bright cortical bony walls (`~0.45–0.60`), flanking the central nasal cavity. Non-sinus inferior slices show the solid U-shaped mandible, bright tooth crowns/roots, and central oral tongue/soft tissue without symmetric enclosed bilateral bony air chambers.

## Exact Forward Pass (`cls-lenet2d-v1`, 261 parameters)
1. `Conv1`: 4 filters of `5 × 5` (`conv1_weights`, shape `[4, 1, 5, 5]`, zero-padding `2`) + `conv1_bias` (`[4]`) → ReLU → `4 × 4` average pooling → `(4, 16, 16)`.
2. `Conv2`: 4 filters of `3 × 3` over 4 input channels (`conv2_weights`, shape `[4, 4, 3, 3]`, zero-padding `1`) + `conv2_bias` (`[4]`) → ReLU → `(4, 16, 16)`.
3. Global pooling: concatenate `[channel_mean(4), channel_max(4)]` → 8-element feature vector.
4. Binary logit: `dot(features, head_weights) + head_bias`. Positive logit predicts maxillary sinus present (`label = 1`).

## Output Format
Return **only** one valid JSON object matching this exact schema (all values finite numbers in `[-1000, 1000]`, no duplicate keys, no comments):

```json
{
  "architecture": "cls-lenet2d-v1",
  "conv1_weights": "shape [4, 1, 5, 5]",
  "conv1_bias": [0.0, 0.0, 0.0, 0.0],
  "conv2_weights": "shape [4, 4, 3, 3]",
  "conv2_bias": [0.0, 0.0, 0.0, 0.0],
  "head_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "head_bias": 0.0
}
```
