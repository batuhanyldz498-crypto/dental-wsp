# DenPAR Segmentation Prompt (`denpar_seg` · `periapical-wsp82-v1`)

You are participating in a blind **Weight-Space Programming (WSP)** experiment for medical image segmentation. Do **not** use web search, code execution, or external tools. You will see no example images.

## Task
Directly write all **82 floating-point parameters** for `periapical-wsp82-v1` to segment the **binary union of teeth** on intra-oral periapical (IOPA) dental radiographs.

## Input & Preprocessing
- Single-channel grayscale periapical radiograph clipped at its 0.5th and 99.5th intensity percentiles and linearly scaled to `[0.0, 1.0]`.
- Aspect-preserving resize centered on a `384 × 384` square canvas; zero-padded margins are excluded during scoring.
- **Critical geometric difference from panoramic X-rays:** Periapical radiographs image 2–4 adjacent teeth at high magnification with **variable framing and orientation** (maxillary teeth point roots upward; mandibular teeth point roots downward). Rely primarily on radiodensity and local multi-scale edge/texture cues (bright homogeneous enamel crowns, intermediate-bright vertical dentin roots with darker central pulp canals, dark periodontal ligament space, surrounded by trabecular alveolar bone and very dark empty air beyond the occlusal crown edge), using only a gentle spatial prior.

## Exact Forward Pass (`periapical-wsp82-v1`, 82 parameters)
1. Start with an aligned-corners bilinearly interpolated `3 × 5` spatial bias grid (`spatial_bias`, shape `[3, 5]`) plus scalar `output_bias`.
2. For mean-pooling scales `s ∈ (1, 4, 16)` (two filters per scale, indices `0–1`, `2–3`, `4–5`):
   - Downsample image by `s × s` average pooling.
   - Apply zero-padded `3 × 3` cross-correlation with `kernels[j]` (shape `[3, 3]`), add `feature_bias[j]`, apply ReLU, upsample by nearest repeat (`s × s`), multiply by `readout[j]`, and add to the logit map.
3. Foreground prediction is `logit >= 0.0` on valid unpadded pixels.

## Output Format
Return **only** one valid JSON object matching this exact schema (all values finite numbers in `[-1000, 1000]`, no duplicate keys, no comments):

```json
{
  "architecture": "periapical-wsp82-v1",
  "kernels": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "... 6 matrices of 3x3 ..."],
  "feature_bias": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "readout": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "spatial_bias": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
  "output_bias": 0.0
}
```
