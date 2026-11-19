"""识别结果一致性比对：验证"划分"不改变识别结论。

背景
----
issue #4 的验收标准之一是 P0/P1/P2 三种划分模式下输出正确。计算划分的前提是：
把算力搬到云端**不能改变识别结果**——否则省下的时延没有意义。本脚本把这件事量化：

1. 以 **P0（全本地、全程无 JPEG 往返）** 为基准真值；
2. 逐个模式跑同一张图，比对车牌号字符串、车牌类型、置信度；
3. 可选扫描 JPEG 压缩质量（95/85/75/60/40），找出**识别开始劣化的临界质量**，
   回答"切片压到多小还能认得出来"。

用法（在 Prj-Python 目录下）::

    python -m partitioning.consistency --images ../resource/images --no-recursive
    python partitioning/consistency.py --images ../resource/images --quality-scan
    python -m partitioning.consistency --images ../resource/images --csv consistency.csv

输出
----
控制台表格 + 可选 CSV。判定口径：车牌号**多重集合**完全相同即记为一致
（多车牌场景与顺序无关）。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

if __package__ in (None, ""):
    # 兼容以脚本方式直接运行：python partitioning/consistency.py ...
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from partitioning.config import PartitionMode
    from partitioning.engine import get_engine
    from partitioning.transport import LocalCloudStub
    from partitioning import pipeline as pipeline_mod
    from partitioning import benchmark as bench
    from partitioning.pipeline import encode_jpeg
else:
    from .config import PartitionMode
    from .engine import get_engine
    from .transport import LocalCloudStub
    from . import pipeline as pipeline_mod
    from . import benchmark as bench
    from .pipeline import encode_jpeg


DEFAULT_MODES: List[str] = ["P0", "P1", "P2", "P3"]
QUALITY_SWEEP: List[int] = [95, 85, 75, 60, 40]

# 车牌类型枚举（与 hyperlpr3.common.typedef 保持一致），仅用于打印
PLATE_TYPE_NAMES = {0: "蓝牌", 1: "黄牌单层", 2: "绿牌", 3: "黄牌双层", -1: "未知"}


# ----------------------------------------------------------------------------
# 结果归一化
# ----------------------------------------------------------------------------
def normalize(plates: Sequence[dict]) -> Tuple[Tuple[str, ...], Tuple[int, ...], float]:
    """把一次识别结果压成可比对的三元组（车牌号多重集 / 类型多重集 / 最高置信度）。"""
    codes = tuple(sorted(str(p.get("plate_code", "")) for p in plates))
    types = tuple(sorted(int(p.get("plate_type", -1)) for p in plates))
    confs = [float(p.get("rec_confidence", 0.0) or 0.0) for p in plates]
    return codes, types, (max(confs) if confs else 0.0)


def same_codes(a: Tuple[str, ...], b: Tuple[str, ...]) -> bool:
    """车牌号多重集是否相同（与车牌顺序无关）。"""
    return Counter(a) == Counter(b)


def fmt_codes(codes: Sequence[str]) -> str:
    return " / ".join(codes) if codes else "(未检出)"


# ----------------------------------------------------------------------------
# 主比对流程
# ----------------------------------------------------------------------------
def run_consistency(images: List[str],
                    modes: Optional[List[str]] = None,
                    quality_scan: bool = False,
                    warmup: int = 1) -> List[dict]:
    modes = modes or DEFAULT_MODES
    engine = get_engine()
    cloud = LocalCloudStub(engine)
    pipe = pipeline_mod.PartitionedPipeline(mode="P0", network="WIFI",
                                            cloud=cloud, simulate=False)

    rows: List[dict] = []
    total = len(images)
    for idx, path in enumerate(images, 1):
        image = cv2.imread(path)
        if image is None:
            print(f"[skip] 读图失败: {path}")
            continue
        print(f"[{idx}/{total}] {os.path.basename(path)} "
              f"({image.shape[1]}x{image.shape[0]})")

        if warmup:  # 预热一次，避免首帧把初始化时间算进去
            pipe.switch_mode("P0")
            pipe.run(image)

        baseline: Optional[Tuple[Tuple[str, ...], Tuple[int, ...], float]] = None
        for mode in modes:
            pipe.switch_mode(mode)
            plates, metrics = pipe.run(image)
            codes, types, conf = normalize(plates)
            if mode.upper().startswith("P0"):
                baseline = (codes, types, conf)

            consistent = "" if baseline is None else (
                "OK" if same_codes(baseline[0], codes) else "DIFF")
            rows.append({
                "image": os.path.basename(path),
                "mode": mode,
                "plate_count": len(plates),
                "plate_codes": fmt_codes(codes),
                "plate_types": ",".join(PLATE_TYPE_NAMES.get(t, str(t)) for t in types),
                "confidence": round(conf, 4),
                "payload_up_kb": round(metrics.payload_up / 1024, 2),
                "consistent_with_p0": consistent,
                "jpeg_quality": 95,
                "is_scan": False,
            })
            print(f"    {mode}: {fmt_codes(codes):<14} "
                  f"conf={conf:.3f}  up={metrics.payload_up / 1024:7.2f} KB  {consistent}")

        # JPEG 质量扫描：只针对 P2（切片上传），看压到多小还能认出来
        if quality_scan:
            rows.extend(_sweep_quality(pipe, image, baseline, os.path.basename(path)))

    return rows


def _sweep_quality(pipe, image, baseline: Optional[tuple], image_name: str) -> List[dict]:
    """把 pipeline 里的 JPEG 编码器临时换成指定质量，跑 P2 看识别是否还一致。"""
    original = pipeline_mod.encode_jpeg
    out: List[dict] = []
    try:
        for q in QUALITY_SWEEP:
            pipeline_mod.encode_jpeg = (lambda quality: (lambda img: original(img, quality)))(q)
            pipe.switch_mode("P2")
            plates, metrics = pipe.run(image)
            codes, types, conf = normalize(plates)
            consistent = "" if baseline is None else (
                "OK" if same_codes(baseline[0], codes) else "DIFF")
            out.append({
                "image": image_name,
                "mode": "P2",
                "plate_count": len(plates),
                "plate_codes": fmt_codes(codes),
                "plate_types": ",".join(PLATE_TYPE_NAMES.get(t, str(t)) for t in types),
                "confidence": round(conf, 4),
                "payload_up_kb": round(metrics.payload_up / 1024, 2),
                "consistent_with_p0": consistent,
                "jpeg_quality": q,
                "is_scan": True,
            })
            print(f"    P2@q{q:<3}: {fmt_codes(codes):<14} conf={conf:.3f}  "
                  f"up={metrics.payload_up / 1024:7.2f} KB  {consistent}")
    finally:
        pipeline_mod.encode_jpeg = original
    return out


# ----------------------------------------------------------------------------
# 汇总与输出
# ----------------------------------------------------------------------------
def print_summary(rows: List[dict]) -> None:
    if not rows:
        return
    print("\n" + "=" * 78)
    print("一致性汇总（以 P0 全本地为基准）")
    print("=" * 78)

    by_mode: Dict[str, List[dict]] = {}
    for r in rows:
        if not r.get("is_scan"):  # 质量扫描行单独统计
            by_mode.setdefault(r["mode"], []).append(r)

    print(f"{'模式':<8}{'样本数':>8}{'一致':>8}{'一致率':>10}{'平均置信度':>12}")
    print("-" * 78)
    for mode in sorted(by_mode):
        sub = by_mode[mode]
        ok = [r for r in sub if r["consistent_with_p0"] == "OK"]
        rate = len(ok) / len(sub) * 100 if sub else 0.0
        conf = sum(r["confidence"] for r in sub) / len(sub) if sub else 0.0
        print(f"{mode:<8}{len(sub):>8}{len(ok):>8}{rate:>9.1f}%{conf:>12.3f}")

    sweep = [r for r in rows if r.get("is_scan")]
    if sweep:
        print("\nJPEG 压缩质量扫描（P2 切片上传）")
        print(f"{'质量':<8}{'样本数':>8}{'一致':>8}{'一致率':>10}{'切片均值':>12}")
        print("-" * 78)
        by_q: Dict[int, List[dict]] = {}
        for r in sweep:
            by_q.setdefault(r["jpeg_quality"], []).append(r)
        for q in sorted(by_q, reverse=True):
            sub = by_q[q]
            ok = [r for r in sub if r["consistent_with_p0"] == "OK"]
            rate = len(ok) / len(sub) * 100 if sub else 0.0
            size = sum(r["payload_up_kb"] for r in sub) / len(sub) if sub else 0.0
            print(f"q{q:<6}{len(sub):>8}{len(ok):>8}{rate:>9.1f}%{size:>10.2f} KB")
    print("=" * 78)


FIELDS = ["image", "mode", "jpeg_quality", "is_scan", "plate_count", "plate_codes",
          "plate_types", "confidence", "payload_up_kb", "consistent_with_p0"]


def write_csv(rows: List[dict], out: str) -> str:
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="consistency",
        description="P0/P1/P2/P3 识别结果一致性比对（issue #4 验收口径）")
    p.add_argument("--images", required=True, help="图片路径或目录")
    p.add_argument("--modes", default="", help="逗号分隔，如 P0,P2；留空表示 P0-P3 全部")
    p.add_argument("--csv", default="", help="结果 CSV 输出路径，留空不落盘")
    p.add_argument("--quality-scan", action="store_true",
                   help="额外扫描 JPEG 质量 95/85/75/60/40 对识别一致率的影响")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--no-recursive", action="store_true",
                   help="只取目录下第一层图片，排除 align/ 等中间产物")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    images = bench.collect_images(args.images, limit=args.limit,
                                  recursive=not args.no_recursive)
    if not images:
        raise SystemExit(f"没找到图片: {args.images}")
    modes = args.modes.split(",") if args.modes else DEFAULT_MODES

    rows = run_consistency(images, modes=modes, quality_scan=args.quality_scan,
                           warmup=args.warmup)
    print_summary(rows)
    if args.csv:
        print(f"\nCSV 已写入: {os.path.abspath(write_csv(rows, args.csv))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
