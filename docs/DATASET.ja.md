# データセット形式

## PASCAL VOC
ディレクトリ構成により自動でアノテーションを生成できます。  
**`JPEGImages` ディレクトリに任意の画像を配置すると、`Annotations` と `SegmentationClass` ディレクトリが自動生成されます。**

```text
EXAMPLE_DIR/
├── JPEGImages/        <-- ここに画像を置いてください
│   ├── image01.jpg
│   ├── image02.jpg
│   └── ...
├── Annotations/       <-- 自動生成されます
│   ├── image01.xml
│   ├── image02.xml
│   └── ...
├── SegmentationClass/ <-- 自動生成（`--segmentation` 指定時のみ）
│   ├── image01.png
│   ├── image02.png
│   └── ...
├── ImagesSets/        <-- 自動生成されます
│   ├── Main
│   │   └── default.txt
│   └── Segmentation  <-- `--segmentation` 指定時のみ作成されます
│       └── default.txt
└── labelmap.txt       <-- CVATでアノテーションする場合は必須
```

#### labelmap.txt（例）
```txt
# label:color_rgb:parts:actions
background:0,0,0::
aeroplane:128,0,0::
bicycle:0,128,0::
bird:128,128,0::
boat:0,0,128::
bottle:128,0,128::
bus:0,128,128::
car:128,128,128::
cat:64,0,0::
chair:192,0,0::
cow:64,128,0::
diningtable:192,128,0::
dog:64,0,128::
horse:192,0,128::
motorbike:64,128,128::
person:192,128,128::
pottedplant:0,64,0::
sheep:128,64,0::
sofa:0,192,0::
train:128,192,0::
tvmonitor:0,64,128::
ignored:224,224,192::
```

## MS COCO
ディレクトリ構成により自動でアノテーションを生成できます。  
**`EXAMPLE_DIR` に任意の画像を配置すると、すべてのアノテーションを含む単一の `annotations.json` ファイルが自動生成されます。**

```text
EXAMPLE_DIR/
├── images/ <-- ここに画像を置いてください
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
│
└── annotations/
    └── instances.json  <-- 自動生成されます

```
