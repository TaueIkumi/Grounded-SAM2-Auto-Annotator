[**🇯🇵 Japanese**](README.ja.md)

# Automated Labeling with Grounded SAM 2

**[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2/tree/main)**

## Overview
This project is a tool for automated annotation on arbitrary images using **Grounded-SAM-2 (GSAM2)** with text prompts.

By inputting predefined class names as text prompts, it automatically generates both **bounding boxes** and **segmentation masks**.

## What is Grounded SAM 2?
Grounded SAM 2 is a system that combines the following two powerful models into a pipeline:

1. **Grounding DINO (Open-Set Object Detection)**
   * Accepts arbitrary text inputs (e.g., "car", "person") and detects corresponding objects in the image with **bounding boxes**.
2. **SAM 2 (Segment Anything Model 2)**
   * Accepts the bounding boxes output by Grounding DINO as prompts (hints) and precisely segments the objects within them as **pixel-level masks**.

This enables the creation of accurate segmentation data simply by "instructing via text."

## Supported Datasets
- [x] COCO bbox
- [x] COCO seg
- [x] Pascal VOC bbox
- [x] Pascal VOC seg

## Installation
Please refer to [INSTALL.md](INSTALL.md).