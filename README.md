# ESA Φ-lab ClearSAR // RFI Detection in Sentinel-1 SAR

This is my run throughout ESA Φ-lab clearSAR challenge. The task was to perform object detection on Sentinel-1 SAR quicklook RGB images to detect Radio-Frequency Interference (RFI). No restrictinos in terms of models or approaches were set, and the metric being evaluated was `COCO mAP@IoU=0.50:0.95`.

<br><br><br>
<p align="center">
  <img src="assets/figures/intro.png" width="90%"/>
  <br><em>Training batch with RFI annotations</em>
</p>
 
<br><br><br>

# Using the model

The best score I managed to get was from single-model fine-tune of `RT-DETR-X`, getting a `0.4682 mAP` on the test dataset. For reference, the top 10 threshold finished at `0.4875`.

<br><br><br>
<p align="center">
  <img src="assets/figures/preds_1.gif" width="40%"/>
  &nbsp;&nbsp;
  <img src="assets/figures/preds_2.gif" width="40%"/>
  <br><em>Some validation images and the prediction with my best attempt</em>
</p>
<br><br><br>

Weights can be found as a release asset for this repo. If you want to try the model you'll need `python>=3.11`. All you have to do is, in an environment of your choice:

```bash
pip install ultralytics

# Download the weights
mkdir checkpoints
curl -L https://github.com/vic-zab/esa_clearsar/releases/latest/download/best.pt -o checkpoints/rtdetr_best.pt
```

Then run inference with:

```python
from ultralytics import RTDETR

model = RTDETR("checkpoints/rtdetr_best.pt")
results = model.predict(source="your_image.png", imgsz=640, conf=0.01, device="cuda")
results[0].show()
```
<br><br><br>

# A little walk through the whole process
I am now going to briefly discuss the development of the whole projects, things I've tried, things that have worked out and things that have not worked out. Notebooks can be found in the `ClearSAR/` folder.
<br><br>

## Exploratory Data Analysis (`EDA_notebook.ipynb`)
Before doing anything, the usual: get to know the data.
We're provided the following *Dataset* from the [eotdl site](https://www.eotdl.com/datasets)

<div align="center">

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 3,154  | 9,288       |
| Test  | 786    | —           |

</div>

<br>
After doing some exploration, the most important insight was the three-column spatial prior: RFI box centers cluster at x≈0.15, 0.50, 0.85. Since the columns are symmetric around x=0.5, horizontal flip is a free augmentation.

<br><br><br>
<p align="center">
  <img src="assets/figures/output3.png" width="70%"/>
</p>
<br><br><br>

## YOLOv8 baseline (`ClearSAR/phase_1.ipynb`)

Established a baseline, trained at 640x640 with letterboxing, reached **0.4285 val mAP@0.50:0.95**
<br><br>

## Architecture Experiments + RT-DETR (`ClearSAR/phase_2.ipynb`)

Two parallel experiments:
- **Exp B:** YOLO with a 4th positional channel encoding x-position. Result: +0.0008 mAP (0.4293).
- **Exp C:** First RT-DETR run. RT-DETR-L, same training setup. Result: **0.4484 val mAP**. Transformer cross-attention exploits the spatial prior naturally without any hand-engineering.
<br><br>

## WBF Ensemble: RT-DETR-L + YOLO (`ClearSAR/phase_3.ipynb`)

Attempted to fuse the two best models via Weighted Boxes Fusion (WBF) at `iou_thr=0.25`. Result: **~0.35–0.38 mAP**.
<br><br>

## RT-DETR-X **(Best Single Model)** (`ClearSAR/phase_4.ipynb`)

Not much to be said. Scaled up to RT-DETR-X. Result: **0.4503 val mAP / 0.4682 test mAP**. Current best.
<br><br>

## RT-DETR-X + extra P2 (`ClearSAR/phase_5.ipynb`)

My intuition was that the thinnest boxes were slipping through because of the feature map downsampling in the FPN, so I tried adding a P2 stride-4 head to RT-DETR-X (76M params, 4-level decoder). Result: **0.3482 val mAP**.