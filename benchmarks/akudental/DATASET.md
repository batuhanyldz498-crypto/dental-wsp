# AKUDENTAL Dataset (2D Panoramic Radiographs)

## Citation & License
- **Authors:** Melih Oz, Aycan Sengul, Mukerrem Hatipoglu, Taner Danisman
- **Paper:** *AKUDENTAL teeth instance segmentation dataset: a cross-dataset analysis*, BMC Oral Health (2025), [https://doi.org/10.1186/s12903-025-07645-0](https://doi.org/10.1186/s12903-025-07645-0)
- **Official Repository:** [https://github.com/melihoz/AKUDENTAL](https://github.com/melihoz/AKUDENTAL)
- **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Annotations Used in `dental-wsp`
- **`akudental_seg` (`pano-wsp82-v1`):** Binary union of the 32 permanent FDI tooth classes (`11–48`, source COCO foreground classes `1–32`) on a `384 × 768` aspect-preserving padded canvas after per-film `0.5th–99.5th` percentile intensity clipping.
- **`akudental_cls` (`cls-lenet2d-v1`):** Binary presence of any radiopaque restorative class (`Implant`, `Crown/Filling`, `Bridge`) on a `64 × 64` grayscale canvas.

## Bring-Your-Own-Data (BYOD) Placement
Place the dataset outside Git (for example under `data/akudental/` or Google Drive `WSP_Dental_Data/AKUDENTAL/`):
- **Official public COCO folder:** `akudental_instances.json` + `images/`
- **Legacy archive format:** `AKUDENTAL_PANORAMIC_INIT_V1.zip` + `split_plan.private.json`

Verify locally without modifying source files:
```bash
dental-wsp verify-dataset akudental --data-dir data/akudental
```
