#import "@preview/cetz:0.3.4": canvas, draw

#set page(width: 22cm, height: 12cm, margin: 1.2cm)
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
  content((x + 0.15, y + h - 0.4), text(size: 8.5pt, weight: "bold", fill: textc)[#title])
}

#align(center)[
  #text(size: 13pt, weight: "bold")[图 2　HyperLPR 推理流水线数据流与 P2 黄金切分点]
]

#align(center)[
  #canvas({
    import draw: *

    // 端侧
    band(0, 6.9, 16.6, 2.5, "端侧 Mobile　·　P0 全本地执行　/　P2 到此为止",
      rgb("#e6f1fb"), rgb("#185fa5"), rgb("#0c447c"))
    node(0.25, 7.35, 2.5, "输入原图", "数百 KB ~ 2 MB", rgb("#ffffff"), rgb("#85b7eb"), rgb("#0c447c"))
    node(3.15, 7.35, 3.0, "Stage 1 目标定位", "19.2 ms · 占 48%", rgb("#ffffff"), rgb("#85b7eb"), rgb("#0c447c"))
    node(6.55, 7.35, 2.0, "四角点", "8 floats", rgb("#ffffff"), rgb("#85b7eb"), rgb("#0c447c"))
    node(8.95, 7.35, 2.7, "Stage 2 几何对齐", "0.9 ms · 占 2%", rgb("#ffffff"), rgb("#85b7eb"), rgb("#0c447c"))
    node(12.05, 7.35, 3.1, "车牌切片", "48×160 · 9.75 KB", rgb("#faeeda"), rgb("#ba7517"), rgb("#854f0b"))

    line((2.75, 8.05), (3.15, 8.05), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)
    line((6.15, 8.05), (6.55, 8.05), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)
    line((8.55, 8.05), (8.95, 8.05), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)
    line((11.65, 8.05), (12.05, 8.05), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)

    // P2 切分：上传
    line((13.6, 7.35), (13.6, 5.95), (1.8, 5.95), (1.8, 2.75),
      mark: (end: ">", size: 0.16pt), stroke: (paint: rgb("#ba7517"), thickness: 0.9pt, dash: "dashed"))
    content((7.7, 6.2), text(size: 8pt, fill: rgb("#854f0b"))[
      P2 切分点：HTTP POST 仅上传 9.75 KB 切片（相比 P1 整图上传削减 97.5% 载荷）
    ])

    // 云端
    band(0, 2.35, 15.2, 2.5, "云端 Server　·　P1 / P2 / P3 在此执行",
      rgb("#eeedfe"), rgb("#534ab7"), rgb("#3c3489"))
    node(0.25, 2.8, 3.1, "接收切片", "JPEG 字节流", rgb("#ffffff"), rgb("#afa9ec"), rgb("#3c3489"))
    node(3.75, 2.8, 3.2, "Stage 3 字符识别", "CRNN + CTC", rgb("#ffffff"), rgb("#afa9ec"), rgb("#3c3489"))
    node(7.35, 2.8, 3.0, "Stage 4 属性分类", "单/双层 · 颜色", rgb("#ffffff"), rgb("#afa9ec"), rgb("#3c3489"))
    node(10.75, 2.8, 4.2, "车牌号 + 类型 + 置信度", "JSON 回传", rgb("#ffffff"), rgb("#afa9ec"), rgb("#3c3489"))

    line((3.35, 3.5), (3.75, 3.5), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)
    line((6.95, 3.5), (7.35, 3.5), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)
    line((10.35, 3.5), (10.75, 3.5), mark: (end: ">", size: 0.14pt), stroke: 0.8pt)

    content((8.3, 1.5), text(size: 8.5pt, fill: rgb("#444441"))[
      端侧仅用约 20 ms 便将 2 MB 原图压缩为不足 10 KB 的中间张量，这就是动态计算划分的收益来源
    ])
  })
]
