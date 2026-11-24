# AKUDENTAL Classification Prompt (`akudental_cls` · `cls-lenet2d-v1`)

You are participating in a blind **Weight-Space Programming (WSP)** experiment mirroring Lindholz et al. (*Weight-Space Programming for Medical Imaging*). Do **not** use web search, code execution, or external tools. You will see no example images.

## Task
Directly write all **261 floating-point parameters** for `cls-lenet2d-v1` to classify whether a dental panoramic radiograph contains **at least one radiopaque dental restoration or implant** (union of *Implant*, *Crown/Filling*, and *Bridge* vs. no annotated restoration).

## Input & Preprocessing
- Grayscale panoramic radiograph clipped at its 0.5th and 99.5th intensity percentiles, scaled to `[0.0, 1.0]`, and aspect-preserving centered on a `64 × 64` canvas.
- Radiographic physics cue: Metallic implants, crowns, bridges, and amalgam/radiopaque fillings exhibit intense X-ray attenuation—appearing as sharp, localized, very bright focal blobs (near `1.0`) with steep contrast against surrounding enamel, dentin, and alveolar bone.

## Exact Forward Pass (`cls-lenet2d-v1`, 261 parameters)
1. `Conv1`: 4 filters of `5 × 5` (`conv1_weights`, shape `[4, 1, 5, 5]`, zero-padding `2`) + `conv1_bias` (`[4]`) → ReLU → `4 × 4` average pooling → `(4, 16, 16)`.
2. `Conv2`: 4 filters of `3 × 3` over 4 input channels (`conv2_weights`, shape `[4, 4, 3, 3]`, zero-padding `1`) + `conv2_bias` (`[4]`) → ReLU → `(4, 16, 16)`.
3. Global pooling: concatenate `[channel_mean(4), channel_max(4)]` → 8-element feature vector.
4. Binary logit: `dot(features, head_weights) + head_bias`. Positive logit predicts restoration/implant present (`label = 1`).

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
