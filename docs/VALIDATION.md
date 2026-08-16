# Validation and inference geometry

## Segmentation checkpoint

The interface uses the fully fine-tuned SigLIP2 Base NaFlex seed-43 checkpoint.
On the fixed 120-image official Kvasir-SEG test split it achieved:

| Metric | Value |
| --- | ---: |
| Mean Dice | 0.902960 |
| Mean IoU | 0.847710 |
| Precision | 0.922011 |
| Recall | 0.919619 |
| Failure rate (Dice < 0.10) | 0.000000 |

The three-seed mean Dice was 0.901334.

## Deployment sanity check

Six curated images, deliberately including difficult regressions, were checked
against their saved seed-43 per-image references:

| Image | Dice | IoU |
| --- | ---: | ---: |
| cju13hp5rnbjx0835bf0jowgx.jpg | 0.9365 | 0.8805 |
| cju14pxbaoksp0835qzorx6g6.jpg | 0.9838 | 0.9681 |
| cju16whaj0e7n0855q7b6cjkm.jpg | 0.7372 | 0.5838 |
| cju32srle1xfq083575i3fl75.jpg | 0.9695 | 0.9409 |
| cju87xn2snfmv0987sc3d9xnq.jpg | 0.7536 | 0.6047 |
| ck2bxw18mmz1k0725litqq2mc.jpg | 0.2315 | 0.1309 |
| Mean | 0.7687 | 0.6848 |

The mixed subset mean is not the official test-set performance. It verifies that
the deployed preprocessing reproduces stored predictions across easy and hard cases.

## Why segmentation uses a square input

Training and evaluation resized every Kvasir-SEG image and mask to 320 x 320.
The NaFlex encoder therefore produced a 20 x 20 grid (400 patch tokens) for the
decoder. Feeding native-aspect images with padding and reshaping them directly
to 20 x 20 scrambles spatial positions. Deployment consequently:

1. resizes the segmentation branch input to 320 x 320;
2. runs the encoder and decoder in the training geometry;
3. interpolates logits back to the uploaded image size;
4. thresholds probabilities using the checkpoint configuration.

This alignment is essential for correct mask placement.

## Scope

- The classification branch predicts only the 23 HyperKvasir benchmark categories.
- Softmax scores are not calibrated clinical probabilities.
- The Kvasir-SEG branch was trained on polyp images and is not a no-polyp detector.
- The classification gate reduces inappropriate segmentation but is not clinically validated.
- Outputs must not be used for diagnosis, treatment, or patient management.
