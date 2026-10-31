#import "@preview/cetz:0.3.4": canvas, draw

#set page(width: 22cm, height: 13cm, margin: 1.2cm)
#set text(font: ("SimSun", "Source Han Serif SC"), size: 10pt)

#let node(x, y, w, title, sub, fillc, strokec, textc) = {
  import draw: *
  rect((x, y), (rel: (w, 1.4)), radius: 4pt, fill: fillc, stroke: 0.6pt + strokec)
  content((x + w / 2, y + 0.92), text(size: 9pt, weight: "bold", fill: textc)[#title])
  content((x + w / 2, y + 0.48), text(size: 7.5pt, fill: textc.lighten(25%))[#sub])
}

#let band(x, y, w, h, title, fillc, strokec, textc) = {
  import draw: *
  rect((x, y), (rel: (w, h)), radius: 8pt, fill: fillc, stroke: 0.5pt + strokec)
  // 标题单独占用色带顶部一行，避免与节点重叠；north-west 锚点保证从框内左上角起排
  content((x + 0.25, y + h - 0.35), text(size: 8.5pt, weight: "bold", fill: textc)[#title],
    anchor: "north-west")
}

#align(center)[
  #text(size: 13pt, weight: "bold")[HyperLPR 推理流水线数据流与候选切分点]
]

#align(center)[
  #canvas({
    import draw: *

    // 端侧
    band(0, 6.6, 16.6, 3.2, "端侧 Mobile　·　Device Tier",
      rgb("#e6f1fb"), rgb("#185fa5"), rgb("#0c447c"))
    node(0.25, 7.0, 2.5, "输入原图", "数百 KB ~ 2 MB", rgb("#ffffff"), rgb("#85b7eb"), rgb("#0c447c"))
    node(3.15, 7.0, 3.0, "Stage 1 目标定位", "80.9 ms · 占 54%", rgb("#ffffff"), rgb("#85b7eb"), rgb("#0c447c"))
    node(6.55, 7.0, 2.0, "四角点", "8 floats", rgb("#ffffff"), rgb("#85b7eb"), rgb("#0c447c"))
    node(8.95, 7.0, 2.7, "Stage 2 几何对齐", "约 1 ms · 占 0.7%", rgb("#ffffff"), rgb("#85b7eb"), rgb("#0c447c"))
    node(12.05, 7.0, 3.1, "车牌切片", "48×160 · 9.75 KB", rgb("#faeeda"), rgb("#ba7517"), rgb("#854f0b"))

    line((2.75, 7.7), (3.15, 7.7), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)
    line((6.15, 7.7), (6.55, 7.7), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)
    line((8.55, 7.7), (8.95, 7.7), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)
    line((11.65, 7.7), (12.05, 7.7), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)

    // 候选切分点：切片上传
    line((13.6, 7.0), (13.6, 5.7), (1.8, 5.7), (1.8, 2.3),
      mark: (end: ">", size: 0.16pt), stroke: (paint: rgb("#ba7517"), thickness: 0.9pt, dash: "dashed"))
    content((7.7, 6.05), text(size: 8pt, fill: rgb("#854f0b"))[
      候选切分点：此后的模块可移至云端，仅上传 9.75 KB 切片（2 MB 原图削减 99.5%，385 KB 图削减 97.5%）
    ])

    // 云端
    band(0, 1.9, 15.2, 3.2, "云端 Server　·　Cloud Tier",
      rgb("#eeedfe"), rgb("#534ab7"), rgb("#3c3489"))
    node(0.25, 2.3, 3.1, "接收切片", "JPEG 字节流", rgb("#ffffff"), rgb("#afa9ec"), rgb("#3c3489"))
    node(3.75, 2.3, 3.2, "Stage 3 字符识别", "68.0 ms · 占 45%", rgb("#ffffff"), rgb("#afa9ec"), rgb("#3c3489"))
    node(7.35, 2.3, 3.0, "Stage 4 属性分类", "1.7 ms · 条件触发", rgb("#ffffff"), rgb("#afa9ec"), rgb("#3c3489"))
    node(10.75, 2.3, 4.2, "车牌号 + 类型 + 置信度", "JSON 回传", rgb("#ffffff"), rgb("#afa9ec"), rgb("#3c3489"))

    line((3.35, 3.0), (3.75, 3.0), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)
    line((6.95, 3.0), (7.35, 3.0), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)
    line((10.35, 3.0), (10.75, 3.0), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)

    content((8.3, 1.35), text(size: 8.5pt, fill: rgb("#444441"))[
      识别阶段的输入仅是这张 9.75 KB 切片而非原图：把它移出端侧既不增加传输量，
      又可省下近一半的端侧算力——这是后续划分研究的关键观察
    ])
    content((8.3, 0.62), text(size: 7pt, fill: rgb("#77776e"))[
      注：节点耗时为 Android 真机（8 核 ARM + MNN）以差分法测得的分解值，合计 149.98 ms；
      桌面代理平台（32 核 x86 + ONNXRuntime）同图合计 92.04 ms，但阶段占比与真机不同（见报告 §6）
    ])
  })
]
