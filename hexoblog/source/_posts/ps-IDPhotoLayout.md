---
title: Photoshop JSX 证件照自动排版脚本
date: 2026-7-12 17:10:00
tags: [photoshop,jsx]
categories: Photoshop
---
<img src="https://sadness96.github.io/images/blog/ps-IDPhotoLayout/IDPhotoLayout.jpg"/>

<!-- more -->
### 简介
使用 Photoshop JSX 脚本实现证件照自动排版打印，运行在 Photoshop 环境中，利用 Photoshop 提供的 JavaScript 自动化接口，实现从照片导入、尺寸调整、相纸设置到最终排版输出的完整流程。

Photoshop → 文件 → 脚本 → 浏览（选择脚本）→ 选择证件照 → 选择排版参数 → 生成

主要功能包括：
- 支持五寸、六寸、七寸相纸以及自定义尺寸
- 支持横向、纵向以及自动选择最佳排列方向
- 支持小一寸、一寸、大一寸、两寸等常用证件照规格
- 支持自定义照片尺寸
- 根据 DPI 自动计算像素尺寸，保证打印精度
- 自动计算一张相纸可容纳的照片数量
- 自动复制并排列照片
- 自动计算排版区域，实现照片整体居中
- 兼容 Photoshop 2020 及以上版本

### 代码
``` jsx
#target photoshop

app.bringToFront();

(function() {

    var paperSize = {
        "五寸": [127, 89],
        "六寸": [152, 102],
        "七寸": [178, 127]
    };

    var photoSize = {
        "小一寸": [22, 32],
        "一寸": [25, 35],
        "大一寸": [33, 48],
        "两寸": [35, 53]
    };

    var file = File.openDialog("选择证件照片", "*.jpg;*.jpeg;*.png;*.bmp");

    if (!file) return;

    var dlg = new Window("dialog", "证件照自动排版");

    dlg.orientation = "column";
    dlg.alignChildren = "left";

    //================
    // 相纸
    //================
    var pPanel = dlg.add("panel", undefined, "相纸设置");

    var pg = pPanel.add("group");

    pg.add("statictext", undefined, "尺寸");

    var paperBox = pg.add("dropdownlist", undefined, ["五寸", "六寸", "七寸", "自定义"]);

    paperBox.selection = 0;

    pg.add("statictext", undefined, "方向");

    var dirBox = pg.add("dropdownlist", undefined, ["纵向", "横向", "容纳最多"]);

    dirBox.selection = 2;

    var psize = pPanel.add("group");

    psize.add("statictext", undefined, "宽");

    var paperW = psize.add("edittext", undefined, "127");

    paperW.characters = 6;

    psize.add("statictext", undefined, "高");

    var paperH = psize.add("edittext", undefined, "89");

    paperH.characters = 6;

    //================
    //照片
    //================
    var iPanel = dlg.add("panel", undefined, "照片尺寸");

    var ig = iPanel.add("group");

    ig.add("statictext", undefined, "尺寸");

    var photoBox = ig.add("dropdownlist", undefined, ["小一寸", "一寸", "大一寸", "两寸", "自定义"]);

    photoBox.selection = 1;

    var isize = iPanel.add("group");

    isize.add("statictext", undefined, "宽");

    var imgW = isize.add("edittext", undefined, "25");

    imgW.characters = 6;

    isize.add("statictext", undefined, "高");

    var imgH = isize.add("edittext", undefined, "35");

    imgH.characters = 6;

    //dpi
    var dg = dlg.add("group");

    dg.add("statictext", undefined, "DPI");

    var dpi = dg.add("edittext", undefined, "300");

    dpi.characters = 6;

    //间距
    var gg = dlg.add("group");

    gg.add("statictext", undefined, "间距");

    var gap = gg.add("edittext", undefined, "3");

    gap.characters = 6;

    var bg = dlg.add("group");

    var ok = bg.add("button", undefined, "生成");

    var cancel = bg.add("button", undefined, "取消");

    function updatePaper() {

        var n = paperBox.selection.text;

        if (n == "自定义") {
            paperW.enabled = true;
            paperH.enabled = true;
        } else {
            paperW.text = paperSize[n][0];
            paperH.text = paperSize[n][1];

            paperW.enabled = false;
            paperH.enabled = false;
        }
    }

    function updatePhoto() {

        var n = photoBox.selection.text;

        if (n == "自定义") {
            imgW.enabled = true;
            imgH.enabled = true;
        } else {
            imgW.text = photoSize[n][0];
            imgH.text = photoSize[n][1];

            imgW.enabled = false;
            imgH.enabled = false;
        }
    }

    paperBox.onChange = updatePaper;
    photoBox.onChange = updatePhoto;

    updatePaper();
    updatePhoto();

    ok.onClick = function() {
        dlg.close(1);
    };

    cancel.onClick = function() {
        dlg.close(0);
    };

    if (dlg.show() != 1) return;

    //================
    // 参数
    //================
    var DPI = parseFloat(dpi.text);
    if (isNaN(DPI)) DPI = 300;

    var SPACE = parseFloat(gap.text);
    if (isNaN(SPACE)) SPACE = 3;

    var PW = parseFloat(paperW.text);
    var PH = parseFloat(paperH.text);

    var IW = parseFloat(imgW.text);
    var IH = parseFloat(imgH.text);

    //方向
    if (dirBox.selection.text == "纵向") {
        if (PW > PH) {
            var t = PW;
            PW = PH;
            PH = t;
        }
    }

    if (dirBox.selection.text == "横向") {
        if (PW < PH) {
            var t2 = PW;
            PW = PH;
            PH = t2;
        }
    }

    function mm2px(v) {
        return Math.round(v * DPI / 25.4);
    }

    var docW = mm2px(PW);
    var docH = mm2px(PH);

    var targetW = mm2px(IW);
    var targetH = mm2px(IH);

    var gapPx = mm2px(SPACE);

    //================
    // 创建文件
    //================
    var src = app.open(file);

    src.activeLayer.copy();

    var doc = app.documents.add(docW, docH, DPI, "证件照排版", NewDocumentMode.RGB, DocumentFill.WHITE);

    doc.paste();

    var first = doc.activeLayer;

    //缩放
    var b = first.bounds;

    var ow = b[2].as("px") - b[0].as("px");
    var oh = b[3].as("px") - b[1].as("px");

    var scale = Math.min(targetW / ow * 100, targetH / oh * 100);

    first.resize(scale, scale, AnchorPosition.MIDDLECENTER);

    //真实大小
    var realW = first.bounds[2].as("px") - first.bounds[0].as("px");

    var realH = first.bounds[3].as("px") - first.bounds[1].as("px");

    var cols = Math.floor((docW + gapPx) / (realW + gapPx));

    var rows = Math.floor((docH + gapPx) / (realH + gapPx));

    //整体居中
    var totalW = cols * realW + (cols - 1) * gapPx;

    var totalH = rows * realH + (rows - 1) * gapPx;

    var startX = (docW - totalW) / 2;

    var startY = (docH - totalH) / 2;

    //================
    //复制
    //================
    for (var y = 0; y < rows; y++) {
        for (var x = 0; x < cols; x++) {
            var layer;

            if (x == 0 && y == 0) {
                layer = first;
            } else {
                layer = first.duplicate();
            }

            var bx = layer.bounds[0].as("px");

            var by = layer.bounds[1].as("px");

            var tx = startX + x * (realW + gapPx);

            var ty = startY + y * (realH + gapPx);

            layer.translate(tx - bx, ty - by);
        }
    }

    src.close(SaveOptions.DONOTSAVECHANGES);

    alert("完成\n" + "数量：" + cols * rows);

})();
```