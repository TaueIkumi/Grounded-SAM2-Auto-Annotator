
## 環境構築
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```
Grounded-SAM-2のモデルのダウンロードなどは上記リポジトリのREADME.mdを参照してください
### Docker
build
```bash
cd Grounded-SAM-2
docker build -t gsam2-base -f Dockerfile .
cd ..

docker build -t custom .
```
run
```
docker run --gpus all -it --rm --net=host --privileged `
  --shm-size=8g `
  -v "${PWD}/Grounded-SAM-2:/home/appuser/Grounded-SAM-2" `
  -v "${PWD}:/home/appuser/workspace" `
  -w /home/appuser/workspace `
  custom
```
Docker環境から抜け出す場合は```exit```と入力する

GPUが使用できるかを確認
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```
`>> True NVIDIA GeForce RTX ****`のように出力されればGPUが使用可能

### Dataset
データセットには[LSMI_dataset](https://github.com/DY112/LSMI-dataset)を用いる。\
githubからもダウンロードできるが、google formなどにメールアドレスなどを入力しないといけないので面倒 \
NASの`data\prj_illuminant_color_estimation\LSMI-dataset`に配置されている(2025/12/06時点では) \
NAS上のデータセットを利用するにはシンボリックリンクとして指定するのが推奨 \
管理者権限でpowershellやvscodeを起動する必要がある \
```bash
# コマンド例(NASの実際のアドレスを記入する必要がある)
New-Item -ItemType SymbolicLink -Path ".\LSMI-dataset" -Value "\\150.89.228.77\shareFiles\data\prj_illuminant_color_estimation\LSMI-dataset"
```

LSMIデータセット内のjpgをまとめる
```bash
# 1. コピー先ディレクトリを作成 (存在しなければ作成)
New-Item -ItemType Directory -Path ".\LSMI-images" -Force | Out-Null

# 2. サブディレクトリを再帰的に検索し、ファイル名でフィルタリングしてコピーを実行
Get-ChildItem -Path ".\LSMI-dataset" -Recurse -Include *.jpg,*.jpeg | 
    Copy-Item -Destination ".\LSMI-images"
```


## issues
```bash
UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
```
とエラーが出る場合はDocker run後に以下コマンドを入力すると修正可能([github-issue](https://github.com/IDEA-Research/Grounded-SAM-2/issues/56#issuecomment-2471647093))
```bash
pip install --no-build-isolation -e Grounded-SAM-2/grounding_dino
```