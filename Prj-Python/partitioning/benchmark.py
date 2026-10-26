"""基准测试矩阵：图像 × 划分策略 × 网络剖面 → CSV。

中期报告要的"不同划分方式 + 不同网络下的性能数据"就是这张表。
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from typing import Iterable, List, Optional, Sequence

import cv2

from .config import PartitionMode
from .pipeline import PartitionedPipeline


DEFAULT_MODES = [m.value for m in PartitionMode]
DEFAULT_NETWORKS = ["5G", "WIFI", "4G", "3G_WEAK", "POOR"]


def collect_images(source: str, limit: Optional[int] = None,
                   recursive: bool = True) -> List[str]:
    """source 可以是单张图片，也可以是目录（默认递归找 jpg/png）。

    ``recursive=False`` 只取该目录下一层图片——仓库 ``resource/images`` 里还放着
    align/、rec_crop/ 这类中间产物，做端侧基线时必须排除。
    """
    if os.path.isfile(source):
        return [source]
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    found = []
    if recursive:
        for root, _, files in os.walk(source):
            for name in files:
                if name.lower().endswith(exts):
                    found.append(os.path.join(root, name))
    else:
        for name in os.listdir(source):
            path = os.path.join(source, name)
            if os.path.isfile(path) and name.lower().endswith(exts):
                found.append(path)
    found.sort()
    return found[:limit] if limit else found


def _summarize(runs: List[dict]) -> dict:
    """同一配置跑多次：数值字段取中位数，抵消 CPU 抖动带来的假结论。"""
    if len(runs) == 1:
        return runs[0]
    numeric = ("image_size_kb", "det_ms", "align_ms", "rec_ms", "cls_ms", "edge_ms",
               "net_up_ms", "cloud_ms", "net_down_ms", "total_ms", "payload_up_kb",
               "payload_down_b", "confidence", "plate_count")
    out = dict(runs[0])
    for key in numeric:
        values = [float(r[key]) for r in runs]
        out[key] = round(statistics.median(values), 3) if key != "plate_count" \
            else int(statistics.median(values))
    # 识别结果取置信度最高的一次，避免中位数把空结果当成代表
    out["plate_code"] = max(runs, key=lambda r: float(r["confidence"]))["plate_code"]
    return out


def _one_run(pipe: PartitionedPipeline, image, image_path: str, mode: str,
             net: str, size_bytes: int) -> dict:
    pipe.switch_network(net)
    plates, metrics = pipe.run(image)
    best = plates[0] if plates else {}
    return {
        "image": os.path.basename(image_path),
        "image_size_kb": round(size_bytes / 1024.0, 2),
        "resolution": f"{image.shape[1]}x{image.shape[0]}",
        "mode": mode,
        "network": net,
        "det_ms": round(metrics.det_ms, 3),
        "align_ms": round(metrics.align_ms, 3),
        "rec_ms": round(metrics.rec_ms, 3),
        "cls_ms": round(metrics.cls_ms, 3),
        "edge_ms": round(metrics.edge_ms, 3),
        "net_up_ms": round(metrics.net_up_ms, 3),
        "cloud_ms": round(metrics.cloud_ms, 3),
        "net_down_ms": round(metrics.net_down_ms, 3),
        "total_ms": round(metrics.total_ms, 3),
        "payload_up_kb": round(metrics.payload_up / 1024.0, 3),
        "payload_down_b": metrics.payload_down,
        "plate_code": best.get("plate_code", ""),
        "confidence": round(float(best.get("rec_confidence", 0.0)), 4),
        "plate_count": metrics.plate_count,
        "repeat": 1,
    }


def run_matrix(image_paths: Sequence[str],
               modes: Iterable[str] = DEFAULT_MODES,
               networks: Iterable[str] = DEFAULT_NETWORKS,
               simulate: bool = True,
               cloud=None,
               warmup: int = 1,
               repeat: int = 3) -> List[dict]:
    """跑完整矩阵，返回每行一条记录的字典列表。

    ``repeat`` 是每个 (图, 策略, 网络) 组合的实测次数，取中位数落表——
    WSL/虚拟机里单次测量的抖动能到 2 倍，直接落表会得出错误结论。
    """
    modes = [PartitionMode.parse(m).value for m in modes]
    networks = list(networks)
    repeat = max(1, int(repeat))
    rows = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[warn] 读图失败，跳过: {image_path}")
            continue
        size_bytes = os.path.getsize(image_path)

        for mode in modes:
            pipe = PartitionedPipeline(mode=mode, cloud=cloud, simulate=simulate)
            for _ in range(warmup):
                pipe.run(image)

            for net in networks:
                runs = [_one_run(pipe, image, image_path, mode, net, size_bytes)
                        for _ in range(repeat)]
                row = _summarize(runs)
                row["repeat"] = repeat
                rows.append(row)
    return rows


FIELDS = ["image", "image_size_kb", "resolution", "mode", "network",
          "det_ms", "align_ms", "rec_ms", "cls_ms", "edge_ms",
          "net_up_ms", "cloud_ms", "net_down_ms", "total_ms",
          "payload_up_kb", "payload_down_b", "plate_code", "confidence",
          "plate_count", "repeat"]


def write_csv(rows: List[dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def print_table(rows: List[dict]) -> None:
    header = (f"{'image':<18}{'mode':<28}{'net':<9}{'edge':>8}{'up':>9}"
              f"{'cloud':>8}{'total':>9}{'payload':>10}{'plates':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['image'][:17]:<18}{r['mode']:<28}{r['network']:<9}"
              f"{r['edge_ms']:>8.1f}{r['net_up_ms']:>9.1f}{r['cloud_ms']:>8.1f}"
              f"{r['total_ms']:>9.1f}{r['payload_up_kb']:>9.2f}K{r['plate_count']:>7}")


def print_summary(rows: List[dict]) -> None:
    """按 (策略, 网络) 聚合出平均端到端时延，这是报告里最常引用的一张表。"""
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["mode"].split("_")[0], r["network"])].append(r["total_ms"])
    print("\n平均端到端时延 (ms)：图 × 策略 × 网络")
    nets = [n for n in DEFAULT_NETWORKS if any(k[1] == n for k in grouped)]
    print(f"{'策略':<8}" + "".join(f"{n:>10}" for n in nets))
    print("-" * (8 + 10 * len(nets)))
    for mode in [m.value.split("_")[0] for m in PartitionMode]:
        if not any(k[0] == mode for k in grouped):
            continue
        line = f"{mode:<8}"
        for net in nets:
            values = grouped.get((mode, net), [])
            line += f"{statistics.mean(values):>10.1f}" if values else f"{'-':>10}"
        print(line)
