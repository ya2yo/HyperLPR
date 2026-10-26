import time
import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
import psutil
import hyperlpr3 as lpr3
from hyperlpr3.config.settings import onnx_runtime_config as ort_cfg, _DEFAULT_FOLDER_
from hyperlpr3.inference.multitask_detect import MultiTaskDetectorORT
from hyperlpr3.inference.recognition import PPRCNNRecognitionORT
from hyperlpr3.inference.classification import ClassificationORT
from hyperlpr3.common.tools_process import get_rotate_crop_image
from hyperlpr3.common.typedef import *

# Suppress onnxruntime warnings
ort.set_default_logger_severity(3)

def profile_stages(image_path: str, rounds: int = 20):
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    img = cv2.imread(image_path)
    img_h, img_w, img_c = img.shape
    raw_size_bytes = os.path.getsize(image_path)
    mem_size_bytes = img.nbytes

    print("=" * 65)
    print(f"HyperLPR Part A 基线性能评测 (Profiling Benchmark)")
    print(f"输入图像: {image_path}")
    print(f"分辨率: {img_w}x{img_h} ({img_c} 通道)")
    print(f"文件大小: {raw_size_bytes / 1024:.2f} KB | 内存原始尺寸: {mem_size_bytes / 1024:.2f} KB")
    print(f"测试轮数: {rounds} 轮 (取平均值与标准差)")
    print("=" * 65)

    folder = _DEFAULT_FOLDER_
    det = MultiTaskDetectorORT(os.path.join(folder, ort_cfg['det_model_path_320x']), input_size=(320, 320))
    rec = PPRCNNRecognitionORT(os.path.join(folder, ort_cfg['rec_model_path']), input_size=(48, 160))
    cls = ClassificationORT(os.path.join(folder, ort_cfg['cls_model_path']), input_size=(96, 96))

    # Warmup
    for _ in range(3):
        outputs = det(img)
        if len(outputs) > 0:
            out = outputs[0]
            landmarks = out[5:13].reshape(4, 2).astype(int)
            pad = get_rotate_crop_image(img, landmarks)
            rec(pad)
            cls(pad)

    det_times = []
    align_times = []
    rec_times = []
    cls_times = []
    patch_sizes = []

    # 系统资源采集（RSS 峰值 + 进程 CPU 占用率）
    proc = psutil.Process()
    cpu_count = psutil.cpu_count() or 1
    rss_peak = proc.memory_info().rss
    cpu_t0 = time.process_time()
    wall_t0 = time.perf_counter()

    for _ in range(rounds):
        # 1. Detector
        t0 = time.perf_counter()
        outputs = det(img)
        t1 = time.perf_counter()
        det_times.append((t1 - t0) * 1000)

        # Assuming single plate for baseline profiling
        if len(outputs) > 0:
            out = outputs[0]
            landmarks = out[5:13].reshape(4, 2).astype(int)
            layer_num = int(out[13])

            # 2. Align (get_rotate_crop_image)
            t2 = time.perf_counter()
            pad = get_rotate_crop_image(img, landmarks)
            t3 = time.perf_counter()
            align_times.append((t3 - t2) * 1000)

            # Record patch size
            _, encoded_patch = cv2.imencode(".jpg", pad)
            patch_sizes.append(len(encoded_patch))

            # 3. Recognizer
            t4 = time.perf_counter()
            if layer_num == DOUBLE:
                h, w, _ = pad.shape
                line = int(h * 0.4)
                code1, conf1 = rec(pad[:line, :])
                code2, conf2 = rec(pad[line:, :])
                plate_code = code1 + code2
            else:
                plate_code, rec_conf = rec(pad)
            t5 = time.perf_counter()
            rec_times.append((t5 - t4) * 1000)

            # 4. Classifier
            t6 = time.perf_counter()
            cls_out = cls(pad)
            t7 = time.perf_counter()
            cls_times.append((t7 - t6) * 1000)
            rss_now = proc.memory_info().rss
            if rss_now > rss_peak:
                rss_peak = rss_now

    avg_det = np.mean(det_times)
    std_det = np.std(det_times)
    avg_align = np.mean(align_times)
    std_align = np.std(align_times)
    avg_rec = np.mean(rec_times)
    std_rec = np.std(rec_times)
    avg_cls = np.mean(cls_times)
    std_cls = np.std(cls_times)
    total_avg = avg_det + avg_align + avg_rec + avg_cls

    avg_patch_size = np.mean(patch_sizes) if patch_sizes else 0

    print("\n【阶段耗时 Breakdown (毫秒)】")
    print(f"1. Detector (目标检测定位):    {avg_det:7.2f} ms (±{std_det:.2f})  [{avg_det/total_avg*100:5.1f}%]")
    print(f"2. Aligner  (ROI透视对齐):     {avg_align:7.2f} ms (±{std_align:.2f})  [{avg_align/total_avg*100:5.1f}%]")
    print(f"3. Recognizer (CTC字符识别):  {avg_rec:7.2f} ms (±{std_rec:.2f})  [{avg_rec/total_avg*100:5.1f}%]")
    print(f"4. Classifier (颜色/版型分类): {avg_cls:7.2f} ms (±{std_cls:.2f})  [{avg_cls/total_avg*100:5.1f}%]")
    print("-" * 65)
    print(f"总计本地端侧平均推理耗时:        {total_avg:7.2f} ms")

    print("\n【数据载荷传输量 (Payload Size) 对比】")
    print(f"输入原图大小:             {raw_size_bytes / 1024:7.2f} KB  (P1 全云端卸载需上传)")
    print(f"车牌矫正切片大小:         {avg_patch_size / 1024:7.2f} KB  (P2 端云协同需上传)")
    reduction = (1 - avg_patch_size / raw_size_bytes) * 100
    print(f"P2 切分点网络载荷削减率:    {reduction:7.2f} % (数据量缩减超 {raw_size_bytes/avg_patch_size:.0f} 倍!)")
    cpu_t1 = time.process_time()
    wall_t1 = time.perf_counter()
    cpu_util = (cpu_t1 - cpu_t0) / (wall_t1 - wall_t0) * 100.0 / cpu_count
    rss_peak_mb = rss_peak / (1024 * 1024)

    print("\n【系统资源占用 (Resource Usage)】")
    print(f"峰值常驻内存 (RSS):        {rss_peak_mb:7.1f} MB")
    print(f"进程 CPU 占用率:           {cpu_util:7.1f} % (占 {cpu_count} 核总 CPU)")

    print("=" * 65)

if __name__ == "__main__":
    profile_stages("../resource/images/test_img.jpg", rounds=20)
