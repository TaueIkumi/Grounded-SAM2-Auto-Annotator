# ANNOTATE BY CVAT

```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
```

```bash
$Env:CVAT_HOST = "host.docker.internal"
$Env:CVAT_VERSION = "v2.12.0"
```

### Build
```bash
docker compose up -d
```

### Register
```bash
# enter docker image first
docker exec -it cvat_server /bin/bash
# then run
python3 ~/manage.py createsuperuser

```


