# Mission AI Service - Random Forest Land Cover

## 1. Luồng xử lý

GeoTIFF không được upload qua API. File được đặt trực tiếp vào file server dùng
chung, sau đó API chỉ nhận đường dẫn tương đối.

```text
Nginx file server/geotiff/input/*.tif
        -> POST /api/inference/landcover/geotiff
        -> PostgreSQL inference_jobs
        -> Redis queue: landcover
        -> mission-landcover-worker
        -> geotiff/results/{job_id}/landcover_prediction.tif
        -> MinIO result.json
        -> MQTT completed/failed
```

## 2. Chuẩn bị thư mục dùng chung

Trên host DEV:

```bash
mkdir -p /home/u01/hoanganh/file_server_test/geotiff/input
mkdir -p /home/u01/hoanganh/file_server_test/geotiff/results
chmod -R 755 /home/u01/hoanganh/file_server_test/geotiff
```

Đặt file đầu vào bằng `cp`, `scp`, đồng bộ NAS hoặc tiến trình khác:

```bash
cp mang_den_2026.tif \
  /home/u01/hoanganh/file_server_test/geotiff/input/
```

Không truyền `/home/...` vào API. Đường dẫn gửi API phải tương đối so với thư
mục `geotiff`, ví dụ `input/mang_den_2026.tif`.

## 3. Cấu hình Nginx file server

Nếu Nginx hiện tại chỉ khai báo riêng `/images/` và `/videos/`, bổ sung:

```nginx
location /geotiff/ {
    alias /home/u01/hoanganh/file_server_test/geotiff/;
    autoindex off;
    sendfile on;
    types {
        image/tiff tif tiff;
    }
    default_type application/octet-stream;
}
```

Kiểm tra và reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

Nếu Nginx chạy bằng Docker, container Nginx cũng phải mount cùng thư mục host.

## 4. Đặt model

```text
app/models/landcover_random_forest.joblib
```

Trong `mission_models`, model của mission land-cover phải có tối thiểu:

```text
model_name    = landcover_random_forest
model_version = 1.0.0
model_path    = /app/models/landcover_random_forest.joblib
is_active     = true
```

`mission_id` gửi API là `missions.code`, tương tự các API inference hiện tại.

Model artifact nên chứa:

```python
{
    "model": random_forest_model,
    "class_names": {
        1: "rung",
        2: "nuoc",
        3: "dat_trong",
        4: "xay_dung",
        5: "nong_nghiep",
    },
    "band_order": ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"],
    "fallback_band_indexes": [1, 2, 3, 4, 5, 6, 7],
}
```

## 5. Build và chạy

Có thể đổi đường dẫn host bằng biến `FILE_SERVER_HOST_DIR`. Nếu không đặt, Compose
dùng `/home/u01/hoanganh/file_server_test`.

```bash
export FILE_SERVER_HOST_DIR=/home/u01/hoanganh/file_server_test
export FILE_SERVER_PUBLIC_URL=http://10.1.53.88:9080
export MINIO_PUBLIC_ENDPOINT=http://10.1.53.88:9210

docker compose build api gpu-worker landcover-worker
docker compose up -d api gpu-worker landcover-worker
docker compose ps
docker compose logs -f landcover-worker
```

## 6. Gọi API

```bash
curl -X POST "http://10.1.53.88:9202/api/inference/landcover/geotiff" \
  -H "Content-Type: application/json" \
  -d '{
    "mission_id": "LANDCOVER",
    "relative_path": "input/mang_den_2026.tif"
  }'
```

Response HTTP 202:

```json
{
  "job_id": "0de7c12f-13fd-46d2-92d8-7c8592df74f4",
  "status": "queued",
  "input_type": "LANDCOVER_GEOTIFF_PATH",
  "status_url": "/api/inference/jobs/0de7c12f-13fd-46d2-92d8-7c8592df74f4",
  "result_url": "/api/inference/jobs/0de7c12f-13fd-46d2-92d8-7c8592df74f4/result"
}
```

Kiểm tra trạng thái:

```bash
curl "http://10.1.53.88:9202/api/inference/jobs/JOB_ID"
```

Lấy kết quả:

```bash
curl "http://10.1.53.88:9202/api/inference/jobs/JOB_ID/result"
```

GeoTIFF đầu ra:

```text
http://10.1.53.88:9080/geotiff/results/JOB_ID/landcover_prediction.tif
```

## 7. Kiểm tra band

Worker ưu tiên description của band trong GeoTIFF. Nếu không có description,
worker lấy các vị trí trong `fallback_band_indexes` của model artifact.

Tất cả ảnh phải dùng cùng phép tiền xử lý như lúc train:

- cùng thứ tự B2, B3, B4, B8, B11, B12, NDVI;
- cùng thang giá trị phản xạ, ví dụ đã chia 10000 hoặc chưa chia;
- cùng công thức NDVI;
- NoData và vùng ngoài AOI được mask đúng.

Nếu model train bằng reflectance 0..1 nhưng ảnh inference còn 0..10000, kết quả
sẽ sai dù API và Docker vẫn chạy thành công.

## 8. File đã bổ sung/thay đổi

```text
app/landcover_inference.py  # đọc theo window và dự đoán Random Forest
app/landcover_tasks.py      # Celery task, DB, MinIO, MQTT
app/main.py                 # endpoint và trả result land-cover
app/schemas.py              # request schema
app/config.py               # FILE_SERVER_GEOTIFF_DIR
app/celery_app.py           # đăng ký task mới
requirements.txt            # rasterio, joblib, scikit-learn
docker-compose.yml          # volume chung và CPU worker riêng
Dockerfile.worker           # model được mount, không COPY model khi build
```
