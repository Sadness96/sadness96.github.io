function imgShow(outerdiv, innerdiv, bigimg, _this) {
    $("body").append('<div id="outerdiv" style="position:fixed;top:0;left:0;background:rgba(0,0,0,0.7);z-index:2;width:100%;height:100%;display:none;"><div id="innerdiv" style="position:absolute;"><img id="bigimg" style="border:5px solid #fff;" src="" /></div></div>')
    // 获取当前点击的pimg元素中的src属性  
    var src = _this.attr("alt");
    // 设置#bigimg元素的src属性
    $(bigimg).attr("src", src);
    // 获取当前点击图片的真实大小，并显示弹出层及大图
    $(bigimg).on('load', function () {
        // 获取当前窗口宽度
        var windowW = $(window).width();
        // 获取当前窗口高度
        var windowH = $(window).height();
        // 获取图片真实宽度
        var realWidth = this.width;
        // 获取图片真实高度
        var realHeight = this.height;
        var imgWidth, imgHeight;
        // 缩放尺寸，当图片真实宽度和高度大于窗口宽度和高度时进行缩放
        var scale = 0.95;
        // 判断图片高度
        if (realHeight > windowH * scale) {
            // 如大于窗口高度，图片高度进行缩放
            imgHeight = windowH * scale;
            // 等比例缩放宽度
            imgWidth = imgHeight / realHeight * realWidth;
            // 如宽度扔大于窗口宽度
            if (imgWidth > windowW * scale) {
                // 再对宽度进行缩放
                imgWidth = windowW * scale;
            }
        } else if (realWidth > windowW * scale) {
            // 如图片高度合适，判断图片宽度
            // 如大于窗口宽度，图片宽度进行缩放
            imgWidth = windowW * scale;
            // 等比例缩放高度
            imgHeight = imgWidth / realWidth * realHeight;
        } else {
            // 如果图片真实高度和宽度都符合要求，高宽不变
            imgWidth = realWidth;
            imgHeight = realHeight;
        }
        // 以最终的宽度对图片缩放
        $(bigimg).css("width", imgWidth);
        // 计算图片与窗口左边距
        var w = (windowW - imgWidth) / 2;
        // 计算图片与窗口上边距
        var h = (windowH - imgHeight) / 2;
        // 设置#innerdiv的top和left属性
        $(innerdiv).css({ "top": h, "left": w });
        // 淡入显示#outerdiv及.pimg
        $(outerdiv).fadeIn("fast");
    })

    // 再次点击淡出消失弹出层
    $(outerdiv).click(function () {
        $(this).fadeOut("fast");
        $("#outerdiv").remove();
    });
}  