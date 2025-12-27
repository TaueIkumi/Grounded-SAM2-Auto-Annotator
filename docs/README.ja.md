# Grounded SAM 2を用いたラベリング

**[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2/tree/main)**

## 概要
本プロジェクトは、**Grounded-SAM-2 (GSAM2)** を用いて、任意の画像データに対してテキストプロンプトによる自動アノテーションを行うツールです。

事前に定義されたクラス名をテキストプロンプトとして入力し、物体検出からセグメンテーションマスクの生成までを自動化しています。

## Grounded SAM 2 とは
Grounded SAM 2 は、以下の2つの強力なモデルをパイプラインとして組み合わせたシステムです。

1.  **Grounding DINO (Open-Set Object Detection)**
    * 任意のテキスト（"car", "person"など）を入力として受け取り、画像内の該当する物体を**バウンディングボックス（矩形）**で検出します。
2.  **SAM 2 (Segment Anything Model 2)**
    * Grounding DINOが出力したバウンディングボックスをプロンプト（ヒント）として受け取り、その内部の物体を**ピクセル単位のマスク**として高精度に切り抜きます。

これにより、「言葉で指示するだけ」で、対象物体の正確なセグメンテーションデータの作成が可能になります。

## 対応データセット
- [x] Coco bbox
- [x] Coco seg
- [x] Pascal VOC bbox
- [x] Pascal VOC seg

## 環境構築
[INSTALL.md](#INSTALL.ja.md)を参照
