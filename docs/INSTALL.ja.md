
## 環境構築
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```
Grounded-SAM-2のモデルのダウンロードなどは[README.ja.md](README.ja.md)を参照してください
### Docker
Build
```bash
cd Grounded-SAM-2
docker build -t gsam2-base -f Dockerfile .
cd ..

docker build -t custom .
```

Run
```
docker run --gpus all -it --rm --net=host --privileged \
  --shm-size=8g \
  -v "${PWD}/Grounded-SAM-2:/home/root/Grounded-SAM-2" \
  -v "${PWD}:/home/root/workspace" \
  -w /home/root/workspace \
  custom
```
Docker環境から抜け出す場合は```exit```と入力する

GPUが使用できるかを確認
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```
`>> True NVIDIA GeForce RTX ****`のように出力されればGPUが使用可能


## トラブルシューティング
###  UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
とエラーが出る場合はDocker run後に以下コマンドを入力すると修正可能([github-issue](https://github.com/IDEA-Research/Grounded-SAM-2/issues/56#issuecomment-2471647093))
```bash
pip install --no-build-isolation -e Grounded-SAM-2/grounding_dino
```

### ModuleNotFoundError: No module named 'grounding_dino'
スクリプト実行時に `grounding_dino` が見つからないというエラーが出る場合は、`PYTHONPATH` を設定する必要があります。
これは Grounded-SAM-2 内部のコードが、特定のディレクトリ構造（フォルダ名を含む絶対/相対インポート）に依存しているためです。

Dockerコンテナ内で以下を実行してパスを通してください：

```bash
export PYTHONPATH=$PYTHONPATH:/home/appuser/workspace/Grounded-SAM-2
```