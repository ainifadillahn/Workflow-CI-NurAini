# Docker Hub Repository

Model AI4I 2020 ini tersedia sebagai Docker image di Docker Hub.

## Docker Hub URL
**https://hub.docker.com/r/ainifadillahn/ai4i2020-model**

## Pull Image
```bash
docker pull ainifadillahn/ai4i2020-model:latest
```

## Run Container
```bash
docker run -p 5001:8080 ainifadillahn/ai4i2020-model:latest
```

Model inference endpoint akan tersedia di: `http://localhost:5001`

## Test Model
```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["feature1", "feature2"], "data": [[1, 2]]}}'
```

## Tags Available
- `latest`: Latest stable version
- `v{run_number}`: Specific build version