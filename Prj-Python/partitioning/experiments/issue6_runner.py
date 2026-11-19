"""Issue #6：全矩阵基准、网络突变和四张图。

在 Prj-Python 目录执行::

    python -m partitioning.experiments.issue6_runner
    python -m partitioning.experiments.issue6_runner --smoke

测的是主机上的 Python 参考仿真：云端用本地等价桩，网络时延按公式估算，
不把结果写成 Android 真机或真实服务器实测。
"""

from __future__ import annotations

import argparse
import csv
import os
import platform

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from partitioning.config import NETWORK_PRESETS, NetworkProfile, PartitionMode
    from partitioning.pipeline import PartitionedPipeline
else:
    from ..config import NETWORK_PRESETS, NetworkProfile, PartitionMode
    from ..pipeline import PartitionedPipeline


# 手机侧能耗只用公开的数量级假设，不是功率计读数。
# 瓦 × 毫秒 = 毫焦。云端计算不记进手机电量，只记手机 CPU 和射频。
P_CPU_W = 2.0
P_RADIO_W = 1.5

STATIC_MODES = [
    PartitionMode.P0_ALL_LOCAL,
    PartitionMode.P1_ALL_CLOUD,
    PartitionMode.P2_EDGE_DET_CLOUD_REC,
    PartitionMode.P3_EDGE_DET_CLOUD_ALIGN_REC,
]
NETWORKS = ["5G", "WIFI", "4G", "3G_WEAK", "POOR"]
GARAGE = NetworkProfile("GARAGE_1MBPS", 1.0, 200.0, jitter_ms=40.0)

MODE_COLOR = {
    "P0": "#4C78A8",
    "P1": "#F58518",
    "P2": "#54A24B",
    "P3": "#E45756",
    "DYNAMIC": "#B279A2",
}
MODE_ORDER = ["P0", "P1", "P2", "P3", "DYNAMIC"]


def reference_dynamic_mode(profile: NetworkProfile) -> PartitionMode:
    """Issue #5 验收里写的对照规则。队长的决策引擎还没合入，突变实验用这条规则做参照。"""
    if profile.bandwidth_mbps >= 80:
        return PartitionMode.P1_ALL_CLOUD
    if profile.bandwidth_mbps >= 10:
        return PartitionMode.P2_EDGE_DET_CLOUD_REC
    return PartitionMode.P0_ALL_LOCAL


def phone_energy_mj(edge_ms: float, net_up_ms: float, net_down_ms: float) -> float:
    return P_CPU_W * edge_ms + P_RADIO_W * (net_up_ms + net_down_ms)


def _repo_python() -> Path:
    return Path(__file__).resolve().parents[2]


def prepare_images(out_dir: Path) -> list[dict]:
    """只用自然场景图，并做出 1080p / 720p / 长边 320 三档。"""
    images_root = _repo_python().parent / "resource" / "images"
    sources = [
        ("test_img", images_root / "test_img.jpg"),
        ("scene2", images_root / "2.jpg"),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for scene, path in sources:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"读图失败: {path}")
        variants = [("native", image)]
        if scene == "test_img":
            variants = [
                ("1080p", image),
                ("720p", cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)),
                ("320", _resize_long_side(image, 320)),
            ]
        for bucket, frame in variants:
            dest = out_dir / f"{scene}_{bucket}.jpg"
            cv2.imwrite(str(dest), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            samples.append({
                "scene": scene,
                "bucket": bucket,
                "path": dest,
                "width": frame.shape[1],
                "height": frame.shape[0],
            })
    return samples


def _resize_long_side(image, long_side: int):
    height, width = image.shape[:2]
    scale = long_side / float(max(height, width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _row_from_run(sample: dict, mode_name: str, chosen: PartitionMode, network: str,
                  repeat_index: int, plates: list, metrics) -> dict:
    best = plates[0] if plates else {}
    energy = phone_energy_mj(metrics.edge_ms, metrics.net_up_ms, metrics.net_down_ms)
    return {
        "scene": sample["scene"],
        "bucket": sample["bucket"],
        "image": sample["path"].name,
        "width": sample["width"],
        "height": sample["height"],
        "resolution": f"{sample['width']}x{sample['height']}",
        "mode": mode_name,
        "chosen_mode": chosen.short,
        "network": network,
        "repeat_index": repeat_index,
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
        "energy_mj": round(energy, 3),
        "plate_code": best.get("plate_code", ""),
        "confidence": round(float(best.get("rec_confidence", 0.0)), 4),
        "plate_count": metrics.plate_count,
    }


def _build_pipes() -> dict:
    print("加载端侧模型（第一次会下载到用户目录）...", flush=True)
    pipes = {}
    for mode in STATIC_MODES:
        pipes[mode] = PartitionedPipeline(mode=mode, simulate=False)
    return pipes


def run_matrix(samples: list[dict], pipes: dict, repeat: int, include_dynamic: bool) -> list[dict]:
    jobs = [(mode.short, mode) for mode in STATIC_MODES]
    if include_dynamic:
        jobs.append(("DYNAMIC", None))
    total = len(samples) * len(jobs) * len(NETWORKS) * repeat
    done = 0
    rows = []
    first = cv2.imread(str(samples[0]["path"]))
    for mode in STATIC_MODES:
        pipes[mode].switch_network("WIFI")
        pipes[mode].run(first)

    for sample in samples:
        image = cv2.imread(str(sample["path"]))
        if image is None:
            print(f"[warn] 读图失败，跳过 {sample['path']}", flush=True)
            continue
        for mode_name, fixed in jobs:
            for network in NETWORKS:
                profile = NETWORK_PRESETS[network]
                chunk = []
                for index in range(1, repeat + 1):
                    chosen = fixed or reference_dynamic_mode(profile)
                    pipe = pipes[chosen]
                    pipe.switch_mode(chosen)
                    pipe.switch_network(network)
                    plates, metrics = pipe.run(image)
                    chunk.append(_row_from_run(sample, mode_name, chosen, network, index, plates, metrics))
                    done += 1
                rows.extend(chunk)
                mean_total = statistics.mean(item["total_ms"] for item in chunk)
                print(f"[{done}/{total}] {sample['path'].name} {mode_name}@{network} "
                      f"mean={mean_total:.1f} ms", flush=True)
    return rows


def run_perturb(sample: dict, pipes: dict, frames: int = 100, switch_at: int = 50) -> list[dict]:
    image = cv2.imread(str(sample["path"]))
    if image is None:
        raise SystemExit(f"突变实验读图失败: {sample['path']}")
    rows = []
    for kind in ("STATIC_P1", "DYNAMIC"):
        for index in range(frames):
            profile = NETWORK_PRESETS["5G"] if index < switch_at else GARAGE
            chosen = PartitionMode.P1_ALL_CLOUD if kind == "STATIC_P1" else reference_dynamic_mode(profile)
            pipe = pipes[chosen]
            pipe.switch_mode(chosen)
            pipe.netsim.switch(profile)
            plates, metrics = pipe.run(image)
            row = _row_from_run(sample, kind, chosen, profile.name, index + 1, plates, metrics)
            row["frame"] = index + 1
            row["condition"] = kind
            rows.append(row)
        print(f"突变序列完成: {kind} {frames} 帧", flush=True)
    return rows


def _mean(values) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def summarize(rows: list[dict]) -> list[dict]:
    """按「图 × 策略 × 网络」汇总重复次数。标准差只描述重复噪声，不把不同分辨率揉在一起。"""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["image"], row["mode"], row["network"])].append(row)
    summary = []
    images = []
    for row in rows:
        if row["image"] not in images:
            images.append(row["image"])
    for image in images:
        for mode in MODE_ORDER:
            for network in NETWORKS:
                chunk = grouped.get((image, mode, network), [])
                if not chunk:
                    continue
                totals = [item["total_ms"] for item in chunk]
                summary.append({
                    "image": image,
                    "scene": chunk[0]["scene"],
                    "bucket": chunk[0]["bucket"],
                    "resolution": chunk[0]["resolution"],
                    "mode": mode,
                    "network": network,
                    "n": len(chunk),
                    "mean_total_ms": round(_mean(totals), 3),
                    "std_total_ms": round(_std(totals), 3),
                    "var_total_ms": round(statistics.pvariance(totals) if len(totals) >= 2 else 0.0, 3),
                    "mean_edge_ms": round(_mean([item["edge_ms"] for item in chunk]), 3),
                    "mean_net_up_ms": round(_mean([item["net_up_ms"] for item in chunk]), 3),
                    "mean_cloud_ms": round(_mean([item["cloud_ms"] for item in chunk]), 3),
                    "mean_payload_up_kb": round(_mean([item["payload_up_kb"] for item in chunk]), 3),
                    "mean_energy_mj": round(_mean([item["energy_mj"] for item in chunk]), 3),
                    "plate_code": max(chunk, key=lambda item: item["confidence"])["plate_code"],
                })
    return summary


def select_image(summary: list[dict], image_name: str) -> list[dict]:
    picked = [item for item in summary if item["image"] == image_name]
    return picked or summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "DengXian"):
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_latency(summary: list[dict], path: Path) -> None:
    plt = _setup_matplotlib()
    import numpy as np

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(NETWORKS))
    width = 0.15
    for offset, mode in enumerate(MODE_ORDER):
        means, errors = [], []
        for network in NETWORKS:
            hit = next((item for item in summary if item["mode"] == mode and item["network"] == network), None)
            means.append(0.0 if hit is None else hit["mean_total_ms"])
            errors.append(0.0 if hit is None else hit["std_total_ms"])
        ax.bar(x + (offset - 2) * width, means, width, yerr=errors, capsize=2,
               label=mode, color=MODE_COLOR[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(NETWORKS)
    ax.set_yscale("log")
    ax.set_ylabel("端到端时延 (ms，对数)")
    ax.set_xlabel("网络剖面")
    ax.set_title("不同网络下各划分策略的端到端时延")
    ax.legend(ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_payload(summary: list[dict], path: Path) -> None:
    plt = _setup_matplotlib()
    by_mode = defaultdict(list)
    for item in summary:
        if item["mode"] == "DYNAMIC":
            continue
        by_mode[item["mode"]].append(item["mean_payload_up_kb"])
    modes = [mode for mode in ("P0", "P1", "P2", "P3") if by_mode[mode]]
    values = [max(_mean(by_mode[mode]), 0.01) for mode in modes]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(modes, values, color=[MODE_COLOR[mode] for mode in modes])
    ax.set_yscale("log")
    ax.set_ylabel("上行载荷 (KB，对数)")
    ax.set_title("各策略上行数据量")
    for mode, value, real in zip(modes, values, [_mean(by_mode[mode]) for mode in modes]):
        ax.text(mode, value * 1.15, f"{real:.2f} KB", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_perturb(rows: list[dict], path: Path, switch_at: int) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for kind, color in (("STATIC_P1", MODE_COLOR["P1"]), ("DYNAMIC", MODE_COLOR["DYNAMIC"])):
        chunk = [item for item in rows if item["condition"] == kind]
        ax.plot([item["frame"] for item in chunk], [item["total_ms"] for item in chunk],
                label=kind, color=color, linewidth=1.6)
    ax.axvline(switch_at + 0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_xlabel("帧序号（按 10 fps 理解）")
    ax.set_ylabel("端到端时延 (ms)")
    ax.set_title("网络从 5G 掉到 1 Mbps 时，固定全云端与参照动态策略")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pareto(summary: list[dict], path: Path) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    markers = {"5G": "o", "WIFI": "s", "4G": "^", "3G_WEAK": "D", "POOR": "v"}
    front = []
    for item in summary:
        dominated = any(
            other["mean_energy_mj"] <= item["mean_energy_mj"]
            and other["mean_total_ms"] <= item["mean_total_ms"]
            and (other["mean_energy_mj"] < item["mean_energy_mj"] or other["mean_total_ms"] < item["mean_total_ms"])
            for other in summary
        )
        if not dominated:
            front.append(item)
    for item in summary:
        ax.scatter(item["mean_energy_mj"], item["mean_total_ms"],
                   c=MODE_COLOR[item["mode"]], marker=markers[item["network"]], s=46, zorder=3)
    if front:
        ordered = sorted(front, key=lambda item: item["mean_energy_mj"])
        ax.plot([item["mean_energy_mj"] for item in ordered],
                [item["mean_total_ms"] for item in ordered],
                color="#222222", linewidth=1, zorder=2)
    for mode, color in MODE_COLOR.items():
        ax.scatter([], [], c=color, marker="o", label=mode)
    for network, marker in markers.items():
        ax.scatter([], [], c="#444444", marker=marker, label=network)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("手机侧估算能耗 (mJ，对数)")
    ax.set_ylabel("端到端时延 (ms，对数)")
    ax.set_title("时延与手机能耗的对照（折线是未被支配的点）")
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _payload_cut(summary: list[dict]) -> tuple[float, float, float]:
    p1 = _mean([item["mean_payload_up_kb"] for item in summary if item["mode"] == "P1"])
    p2 = _mean([item["mean_payload_up_kb"] for item in summary if item["mode"] == "P2"])
    ratio = (p1 - p2) / p1 * 100.0 if p1 else 0.0
    return p1, p2, ratio


def write_report(path: Path, summary: list[dict], perturb_rows: list[dict],
                 repeat: int, switch_at: int, sample_note: str) -> None:
    p1, p2, ratio = _payload_cut(summary)
    lines = [
        "# Issue #6 实验记录（主机参考仿真）",
        "",
        f"- 日期：2026-11-19",
        f"- 机器：{platform.platform()} / {platform.processor() or 'unknown CPU'}",
        f"- Python：{platform.python_version()}",
        "- 云端：`LocalCloudStub`，推理在本机完成，没有调用队员 3 尚未合并的 FastAPI。",
        "- 网络：按 `T = RTT/2 + 载荷比特 / 带宽` 估算，并带剖面里的抖动；没有真实 sleep，也不是手机真网。",
        "- 图片：`test_img.jpg` 的 1080p、720p、长边 320，外加自然场景 `2.jpg`。中间产物 align/rec_crop 没有混入。",
        f"- 重复：每个「图 × 策略 × 网络」{repeat} 次，表内是均值和样本标准差。",
        "- 动态策略：Issue #5 验收里的对照规则（5G 用 P1，Wi-Fi/4G 用 P2，弱网退回 P0）。队长的决策引擎还没合入仓库。",
        f"- 能耗：手机 CPU {P_CPU_W:.1f} W × 端侧毫秒 + 射频 {P_RADIO_W:.1f} W × 传输毫秒，单位毫焦。这是数量级假设，不是功率计。",
        "",
        sample_note,
        "",
        "## 平均端到端时延，test_img 1080p (ms)",
        "",
        "| 策略 | " + " | ".join(NETWORKS) + " |",
        "| :--- | " + " | ".join("---:" for _ in NETWORKS) + " |",
    ]
    for mode in MODE_ORDER:
        cells = []
        for network in NETWORKS:
            hit = next((item for item in summary if item["mode"] == mode and item["network"] == network), None)
            cells.append("—" if hit is None else f"{hit['mean_total_ms']:.1f} ± {hit['std_total_ms']:.1f}")
        lines.append("| " + mode + " | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 上行载荷",
        "",
        f"P1 平均上行 {p1:.2f} KB，P2 平均上行 {p2:.2f} KB，P2 比 P1 少传 {ratio:.1f}%。",
        "",
        "## 网络突变",
        "",
        f"同一张 1080p 图连续 {max((item['frame'] for item in perturb_rows), default=0)} 帧。"
        f"前 {switch_at} 帧为 5G，之后切到 1 Mbps / RTT 200 ms。",
        "固定 P1 在掉速后仍上传整图；参照动态策略改走全本地 P0。",
        "",
    ])
    if perturb_rows:
        for kind in ("STATIC_P1", "DYNAMIC"):
            before = [item["total_ms"] for item in perturb_rows if item["condition"] == kind and item["frame"] <= switch_at]
            after = [item["total_ms"] for item in perturb_rows if item["condition"] == kind and item["frame"] > switch_at]
            lines.append(f"- {kind}：掉速前均值 {_mean(before):.1f} ms，掉速后均值 {_mean(after):.1f} ms。")
    lines.extend([
        "",
        "## 图",
        "",
        "- `fig1_latency.png`：五种策略在五档网络下的端到端时延，纵轴是对数，误差线是 10 次重复的样本标准差。",
        "- `fig2_payload.png`：上行载荷对数坐标。P0 真实值是 0，图上用 0.01 KB 占位才能取对数。",
        "- `fig3_perturb.png`：5G 掉到 1 Mbps 的逐帧时延。",
        "- `fig4_pareto.png`：时延对手机估算能耗，两轴都是对数。折线串起没有被别的点同时压过时延和能耗的配置。",
        "",
        "## 复现",
        "",
        "```bash",
        "cd Prj-Python",
        "python -m partitioning.experiments.issue6_runner",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Issue #6 主机参考仿真")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--switch-at", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="只跑一张图、两个策略、一档网络，确认环境")
    parser.add_argument("--skip-perturb", action="store_true")
    parser.add_argument("--from-raw", action="store_true", help="用已有 raw.csv / perturb.csv 重算表和图")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1] / "results" / "issue6"
    if args.from_raw:
        return _rebuild_from_raw(root, args.repeat, args.switch_at)

    samples = prepare_images(root / "images")
    if args.smoke:
        samples = samples[:1]
        args.repeat = 1
        args.skip_perturb = True
    pipes = _build_pipes()
    if args.smoke:
        rows = _smoke_rows(samples, pipes)
    else:
        rows = run_matrix(samples, pipes, repeat=max(1, args.repeat), include_dynamic=True)
    summary = summarize(rows) if not args.smoke else []
    perturb_rows = []
    native = next(item for item in samples if item["scene"] == "test_img" and item["bucket"] in ("1080p", "native"))
    if not args.skip_perturb:
        perturb_rows = run_perturb(native, pipes, frames=args.frames, switch_at=args.switch_at)

    write_csv(root / "raw.csv", rows)
    if summary:
        write_csv(root / "summary.csv", summary)
    if perturb_rows:
        write_csv(root / "perturb.csv", perturb_rows)
        plot_perturb(perturb_rows, root / "fig3_perturb.png", args.switch_at)
    note = "本次是冒烟，只验证模型能跑，不写入中期结论。" if args.smoke else \
        "主表和图用 test_img 的 1080p。每张图各自的均值在 summary.csv，没有把不同分辨率揉进同一个标准差。"
    if summary:
        focus = select_image(summary, "test_img_1080p.jpg")
        plot_latency(focus, root / "fig1_latency.png")
        plot_payload(focus, root / "fig2_payload.png")
        plot_pareto(focus, root / "fig4_pareto.png")
        doc = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_ISSUE6.md"
        write_report(doc, focus, perturb_rows, args.repeat, args.switch_at, note)
        print(f"报告: {doc}", flush=True)
    print(f"结果目录: {root}", flush=True)
    return 0


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in ("scene", "bucket", "image", "resolution", "mode", "chosen_mode", "network", "plate_code", "condition"):
                continue
            if value == "":
                continue
            row[key] = float(value) if "." in value else int(value)
    return rows


def _rebuild_from_raw(root: Path, repeat: int, switch_at: int) -> int:
    rows = _read_csv(root / "raw.csv")
    perturb_path = root / "perturb.csv"
    perturb_rows = _read_csv(perturb_path) if perturb_path.exists() else []
    summary = summarize(rows)
    write_csv(root / "summary.csv", summary)
    focus = select_image(summary, "test_img_1080p.jpg")
    plot_latency(focus, root / "fig1_latency.png")
    plot_payload(focus, root / "fig2_payload.png")
    plot_pareto(focus, root / "fig4_pareto.png")
    if perturb_rows:
        plot_perturb(perturb_rows, root / "fig3_perturb.png", switch_at)
    note = "主表和图用 test_img 的 1080p。每张图各自的均值在 summary.csv，没有把不同分辨率揉进同一个标准差。"
    doc = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_ISSUE6.md"
    write_report(doc, focus, perturb_rows, repeat, switch_at, note)
    print(f"报告: {doc}", flush=True)
    print(f"结果目录: {root}", flush=True)
    return 0


def _smoke_rows(samples, pipes) -> list[dict]:
    image = cv2.imread(str(samples[0]["path"]))
    rows = []
    for mode in (PartitionMode.P0_ALL_LOCAL, PartitionMode.P2_EDGE_DET_CLOUD_REC):
        pipe = pipes[mode]
        pipe.switch_network("WIFI")
        plates, metrics = pipe.run(image)
        rows.append(_row_from_run(samples[0], mode.short, mode, "WIFI", 1, plates, metrics))
        print(f"smoke {mode.short} total={metrics.total_ms:.1f} ms plates={metrics.plate_count}", flush=True)
    return rows


if __name__ == "__main__":
    sys.exit(main())
