# ANNOTATE BY CVAT

`Grounded-SAM 2` で自動生成したアノテーションを \
`CVAT (Computer Vision Annotation Tool)` にインポートしブラウザ上で確認・修正するためのガイドです。
### CVATサーバーの構築
まずはアノテーションを管理するサーバー（CVAT）を起動します。
#### リポジトリの取得
```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
```
### 環境変数の設定
```bash
export CVAT_HOST="localhost"
export CVAT_VERSION="v2.12.0"
```
### ビルドと起動
```bash
docker compose up -d
```

### アカウント登録
実行スクリプトで使用するユーザーを作成します。
```bash
# enter docker image first
docker exec -it cvat_server /bin/bash
# then run
python3 ~/manage.py createsuperuser

cd ..
```

### GroundedSAM2の環境構築
アノテーションをアップロードする側の環境構築です。詳細は [INSTALL.ja.md](INSTALL.ja.md) を参照してください。

### 実行
Pythonスクリプトを使用して、ローカルのデータセットをCVATに転送します。
```bash
python -m src.annotate \
  -u "http://localhost:8080" \
  -U "your_username" \
  -P "your_password" \
  --format your-format \
  --dataset-dir DATASET-dir/ \
  --task-name "example_task" \
  --labels Alpha Beta Charlie Delta Echo Foxtrot
```

### CVAT上での作業手順
1. アクセス: ブラウザで `http://localhost:8080/tasks` にアクセスします。\
2. タスクの選択: 作成されたタスク（task-name で指定したもの）の Open をクリック。
3. ジョブの開始: 画面下部の Jobs リストにあるリンクをクリックするとエディタが開きます。
4. 修正: AIが作成した枠をマウスで微調整したり、誤検出を削除したりします。
5. 保存: 左上の [Save] アイコン（または Ctrl+S）を忘れずに押してください。