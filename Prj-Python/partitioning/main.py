"""命令行入口：单图演示 / 全矩阵基准测试。

两种调用方式都可以（在 Prj-Python 目录下执行）::

    python -m partitioning.main demo --image resource/images/1.jpg --mode P2 --network 4G
    python partitioning/main.py  demo --image resource/images/1.jpg --mode P2 --network 4G

    python -m partitioning.main bench --images resource/images --out benchmark_results.csv
    python partitioning/main.py  bench --images resource/images --out benchmark_results.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

# 兼容"以脚本方式直接运行"(python partitioning/main.py ...)：
# 此时 __package__ 为空，相对导入会失败，需先把 Prj-Python 目录加进 sys.path 再走绝对导入。
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from partitioning.config import PartitionMode, NETWORK_PRESETS, CLOUD_ENDPOINT
    from partitioning.pipeline import PartitionedPipeline
    from partitioning.transport import build_cloud
    from partitioning import benchmark as bench
else:
    from .config import PartitionMode, NETWORK_PRESETS, CLOUD_ENDPOINT
    from .pipeline import PartitionedPipeline
    from .transport import build_cloud
    from . import benchmark as bench


def _load_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"读图失败: {path}")
    return image


def cmd_demo(args: argparse.Namespace) -> int:
    cloud = build_cloud(args.endpoint, offline=args.offline)
    image = _load_image(args.image)

    pipe = PartitionedPipeline(mode=args.mode, network=args.network, cloud=cloud,
                               simulate=not args.no_sleep)
    plates, metrics = pipe.run(image)

    print("=" * 72)
    print(f"输入: {args.image}  ({image.shape[1]}x{image.shape[0]}, "
          f"{os.path.getsize(args.image) / 1024:.1f} KB)")
    print(f"策略: {metrics.mode}    网络: {pipe.netsim.describe()}")
    print(f"云端: {cloud!r}")
    print("-" * 72)
    print(f"  端侧耗时   {metrics.edge_ms:8.2f} ms   "
          f"(det {metrics.det_ms:.2f} / align {metrics.align_ms:.2f} / "
          f"rec {metrics.rec_ms:.2f} / cls {metrics.cls_ms:.2f})")
    print(f"  上行传输   {metrics.net_up_ms:8.2f} ms   载荷 {metrics.payload_up / 1024:.2f} KB")
    print(f"  云侧耗时   {metrics.cloud_ms:8.2f} ms")
    print(f"  下行传输   {metrics.net_down_ms:8.2f} ms   载荷 {metrics.payload_down} B")
    print(f"  端到端     {metrics.total_ms:8.2f} ms")
    print("-" * 72)
    for i, plate in enumerate(plates, 1):
        print(f"  <{i}> {plate.get('plate_code')}  conf={float(plate.get('rec_confidence', 0)):.3f}  "
              f"type={plate.get('plate_type')}")
    if not plates:
        print("  (未识别到车牌)")
    print("=" * 72)
    if args.json:
        print(json.dumps({"plates": plates, "metrics": metrics.as_dict()}, ensure_ascii=False, indent=2))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    cloud = build_cloud(args.endpoint, offline=args.offline)
    images = bench.collect_images(args.images, limit=args.limit,
                                  recursive=not args.no_recursive)
    if not images:
        raise SystemExit(f"没找到图片: {args.images}")
    modes = args.modes.split(",") if args.modes else bench.DEFAULT_MODES
    networks = args.networks.split(",") if args.networks else bench.DEFAULT_NETWORKS

    print(f"矩阵: {len(images)} 图 × {len(modes)} 策略 × {len(networks)} 网络 "
          f"× {args.repeat} 次 = {len(images) * len(modes) * len(networks) * args.repeat} 次")
    rows = bench.run_matrix(images, modes=modes, networks=networks,
                            simulate=not args.no_sleep, cloud=cloud,
                            warmup=args.warmup, repeat=args.repeat)
    out = bench.write_csv(rows, args.out)
    bench.print_table(rows)
    bench.print_summary(rows)
    print(f"\nCSV 已写入: {os.path.abspath(out)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="partitioning", description="HyperLPR 端云计算划分实验工具")
    parser.add_argument("--endpoint", default=CLOUD_ENDPOINT, help="云端 FastAPI 地址")
    parser.add_argument("--offline", action="store_true", help="不连云，用本地桩做等价推理")
    parser.add_argument("--no-sleep", action="store_true", help="只估算不真实 sleep（快速扫参数）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="单张图跑一次，输出三段式分解")
    p_demo.add_argument("--image", required=True)
    p_demo.add_argument("--mode", default=PartitionMode.P2_EDGE_DET_CLOUD_REC,
                        type=PartitionMode.parse, choices=list(PartitionMode),
                        metavar="{P0,P1,P2,P3}")
    p_demo.add_argument("--network", default="WIFI", choices=list(NETWORK_PRESETS.keys()))
    p_demo.add_argument("--json", action="store_true", help="额外输出 JSON")
    p_demo.set_defaults(func=cmd_demo)

    p_bench = sub.add_parser("bench", help="跑全矩阵并导出 CSV")
    p_bench.add_argument("--images", required=True, help="图片路径或目录")
    p_bench.add_argument("--out", default="benchmark_results.csv")
    p_bench.add_argument("--modes", default="", help="逗号分隔，如 P0,P2；留空表示全部")
    p_bench.add_argument("--networks", default="", help="逗号分隔，留空表示全部")
    p_bench.add_argument("--limit", type=int, default=None)
    p_bench.add_argument("--warmup", type=int, default=1)
    p_bench.add_argument("--repeat", type=int, default=3, help="每个组合实测次数，取中位数")
    p_bench.add_argument("--no-recursive", action="store_true",
                         help="只取目录下第一层图片，排除 align/ 等中间产物")
    p_bench.set_defaults(func=cmd_bench)

    # 顶层/子命令都能写 --no-sleep、--offline：用 SUPPRESS 避免子命令默认值覆盖顶层传参
    for sub_parser in (p_demo, p_bench):
        sub_parser.add_argument("--no-sleep", action="store_true", default=argparse.SUPPRESS,
                                help="只估算不真实 sleep（快速扫参数）")
        sub_parser.add_argument("--offline", action="store_true", default=argparse.SUPPRESS,
                                help="不连云，用本地桩做等价推理")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
