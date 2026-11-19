# 端云接口规约（队员 2 → 队员 3）

> 目的：让云端 FastAPI 微服务（issue #3）与端侧流水线（issue #4）在互不知晓实现的情况下可并行开发。
> 云端未就绪前，端侧用 `LocalCloudStub` 跑等价本地推理，接口语义完全一致，联调时只需换 `--endpoint`。

## 1. 中间张量定义

| 项 | 值 |
|---|---|
| 名称 | 车牌切片（plate patch） |
| 产生位置 | 端侧 Stage 2 `get_rotate_crop_image()` 之后 |
| 形状 | `48 × 160 × 3`，uint8，**BGR**（OpenCV 原生顺序，不要转 RGB） |
| 编码 | JPEG，`quality=95`（可调） |
| 体积 | 约 9.75 KB（实测区间待补，会随车牌纹理浮动） |
| 附带元数据 | `layer_num`：0 = 单层，1 = 双层（决定云端是否分段识别） |

## 2. HTTP 接口

统一前缀：`http://<host>:<port>`，超时 10 s，`multipart/form-data`。

### 2.1 `GET /ping`

- 入参：`?n=<int>` 探针字节数（默认 1024）
- 出参：`{"echo_bytes": <int>, "server_time_ms": <float>}`
- 用途：RTT 与吞吐探针，端侧据此校准 `NetworkSimulator`

### 2.2 `POST /infer/full`（服务 P1）

- 字段：`file` = 整帧 JPEG 字节流
- 出参：

```json
{"plates": [
  {"plate_code": "京A12345",
   "rec_confidence": 0.97,
   "plate_type": 2,
   "det_bound_box": [x1, y1, x2, y2],
   "det_confidence": 0.93}
]}
```

### 2.3 `POST /infer/rec_cls`（服务 P2 / P3）

- 字段：`file` = 切片 JPEG 字节流；`layer_num` = 0 或 1（表单字段，字符串）
- 出参：同上的 `plates` 数组；`det_bound_box` 可为空数组（端侧已知）
- 云端内部：Stage 3 字符识别 → Stage 4 属性分类 → 类型解析

## 3. 端侧调用示例

```python
from partitioning import PartitionedPipeline

pipe = PartitionedPipeline(mode="P2")           # 默认走 LocalCloudStub
pipe.cloud = CloudClient("http://192.168.1.20:8000")   # 云端就绪后换真客户端
plates, metrics = pipe.run(image)
print(metrics.summary())
```

命令行：

```bash
cd Prj-Python
python -m partitioning.main demo --image resource/images/1.jpg --mode P2 --network 4G \
    --endpoint http://192.168.1.20:8000
```

## 4. 一致性验收

- 同一张图在 P0 / P1 / P2 下，`plate_code` 字符串必须完全相同；
- 若 JPEG 压缩导致 P2 结果漂移，降 quality 重测并记录临界值；
- 端侧日志需输出三段式分解：端侧耗时 / 上行传输 / 云侧耗时 / 下行传输。
