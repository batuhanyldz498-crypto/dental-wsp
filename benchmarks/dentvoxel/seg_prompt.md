# DentVoxel 3D CBCT Segmentation Prompt (`dentvoxel_seg` · `cbct-wsp82-v1`)

You are participating in a blind **Weight-Space Programming (WSP)** experiment for 3D volumetric medical image segmentation. Do **not** use web search, code execution, or external tools. You will see no example scans.

## Task
Directly write all **82 floating-point parameters** for `cbct-wsp82-v1` to segment the **3D binary union of natural permanent teeth (32 FDI teeth)** on large field-of-view dental CBCT scans.

## Input & Preprocessing
- 3D CBCT volume in RAS+ canonical orientation reordered to `(Z/inferior-to-superior, Y/posterior-to-anterior, X/left-to-right)`, resampled and centered onto a `96 × 128 × 128` canvas; zero-padded outer margins are excluded during scoring.
- Intensity scaling: dental Hounsfield window `[-1000, +3000]` HU clipped and linearly scaled to `[0.0, 1.0]` (note: air ≈ `0.00`, soft tissue ≈ `0.25`, trabecular bone ≈ `0.30–0.40`, cortical bone ≈ `0.45–0.60`, dentin ≈ `0.55–0.70`, enamel ≈ `0.75–1.00`).
- Anatomical layout: teeth sit in the anterior-inferior-to-mid central dental arch, exhibiting the highest radiodensity in the craniofacial volume (especially enamel crowns), whereas the posterior cranial base/cervical spine sit posteriorly (`low Y`) and maxillary sinuses sit superiorly (`high Z`) as dark air cavities (`~0.0`).

## Exact Forward Pass (`cbct-wsp82-v1`, 82 parameters)
1. Start with an aligned-corners trilinearly interpolated `3 × 3 × 3` spatial bias grid (`spatial_bias`, shape `[3, 3, 3]` indexed `[z, y, x]`) plus scalar `output_bias`.
2. For isotropic 3D mean-pooling scales `s ∈ (1, 2, 4)` (two filters per scale, indices `0–1`, `2–3`, `4–5`):
   - Downsample volume by `s × s × s` average pooling.
   - Apply a zero-padded **7-tap 3D 6-connected stencil** `stencils[j]` ordered `[center, -z, +z, -y, +y, -x, +x]`, add `feature_bias[j]`, apply ReLU, upsample by nearest 3D repeat (`s × s × s`), multiply by `readout[j]`, and add to the 3D logit volume.
3. Foreground prediction is `logit >= 0.0` on valid unpadded voxels.

## Output Format
Return **only** one valid JSON object matching this exact schema (all values finite numbers in `[-1000, 1000]`, no duplicate keys, no comments):

```json
{
  "architecture": "cbct-wsp82-v1",
  "stencils": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "... 6 rows of 7 floats ..."],
  "feature_bias": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "readout": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "spatial_bias": "shape [3, 3, 3]",
  "output_bias": 0.0
}
```
