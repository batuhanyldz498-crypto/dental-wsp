# AKUDENTAL Segmentation Prompt (`akudental_seg` · `pano-wsp82-v1`)

You are participating in a blind **Weight-Space Programming (WSP)** experiment for medical image segmentation. Do **not** use web search, code execution, or external tools. You will see no example images.

## Task
Directly write all **82 floating-point parameters** for `pano-wsp82-v1` to segment the **binary union of natural permanent teeth (FDI 11–48)** on adult dental panoramic radiographs (orthopantomograms).

## Input & Preprocessing
- Single-channel grayscale panoramic radiograph clipped at its 0.5th and 99.5th intensity percentiles and linearly scaled to `[0.0, 1.0]`.
- Aspect-preserving resize centered on a `384 × 768` canvas (height 384, width 768); zero-padded margins are excluded during scoring.
- Teeth (enamel crowns and dentin roots) appear relatively radiopaque (bright) along a central curved upper and lower dental arch occupying roughly the middle horizontal band and central ~60% width of the panoramic field, flanked by mandibular rami, condyles, cervical spine superposition, and dark airway/sinus regions.

## Exact Forward Pass (`pano-wsp82-v1`, 82 parameters)
1. Start with an aligned-corners bilinearly interpolated `3 × 5` spatial bias grid (`spatial_bias`, shape `[3, 5]`) plus scalar `output_bias`.
2. For mean-pooling scales `s ∈ (1, 4, 16)` (two filters per scale, indices `0–1`, `2–3`, `4–5`):
   - Downsample image by `s × s` average pooling.
   - Apply zero-padded `3 × 3` cross-correlation with `kernels[j]` (shape `[3, 3]`), add `feature_bias[j]`, apply ReLU, upsample by nearest repeat (`s × s`), multiply by `readout[j]`, and add to the logit map.
3. Foreground prediction is `logit >= 0.0` on valid unpadded pixels.

## Output Format
Return **only** one valid JSON object matching this exact schema (all values finite numbers in `[-1000, 1000]`, no duplicate keys, no comments):

```json
{
  "architecture": "pano-wsp82-v1",
  "kernels": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "... 6 matrices of 3x3 ..."],
  "feature_bias": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "readout": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "spatial_bias": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
  "output_bias": 0.0
}
```
