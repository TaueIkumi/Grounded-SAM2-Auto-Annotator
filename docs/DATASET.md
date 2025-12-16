# DATASET FORMAT

### PASCAL VOC
The directory structure allows for automatic annotation generation.  \
**Placing arbitrary images in the `JPEGImages` directory will automatically generate the `Annotations` and `SegmentationClass` directories.**

```text
EXAMPLE_DIR/
├── JPEGImages/        <-- Place your images here
│   ├── image01.jpg
│   ├── image02.jpg
│   └── ...
├── Annotations/       <-- Auto-generated
│   ├── image01.xml
│   ├── image02.xml
│   └── ...
├── SegmentationClass/ <-- Auto-generated (created only with --segmentation)
│   ├── image01.png
│   ├── image02.png
│   └── ...
└── ImagesSets/ <-- Auto-generated
    ├── Main
    │   └── default.txt
    └── Segmentation        <-- Created only with --segmentation
        └── default.txt

```

### MS COCO
The directory structure allows for automatic annotation generation.  
**Placing arbitrary images in the `Input Directory` will automatically generate a single `annotations.json` file containing all annotations.**

```text
EXAMPLE_DIR/
├── images/ <-- Place your images here
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
│
└── annotations/
    └── instances.json  <-- Auto-generated

```