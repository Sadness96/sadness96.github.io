---
title: 黑龙江 - 哈尔滨
date: 2023-10-06 21:45:00
tags: [photo]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Harbin/harbinlogo.jpg"/>

<!-- more -->
<ul class="grid effect-1" id="grid">

</ul>

<link rel="stylesheet" type="text/css" href="/blog/lib/masonry/default.css" />
<link rel="stylesheet" type="text/css" href="/blog/lib/masonry/component.css" />
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="/blog/lib/masonry/modernizr.custom.js"></script>
<script src="/blog/lib/masonry/masonry.pkgd.min.js"></script>
<script src="/blog/lib/masonry/imagesloaded.pkgd.min.js"></script>
<script src="/blog/lib/masonry/classie.js"></script>
<script src="/blog/lib/masonry/AnimOnScroll.js"></script>
<script src="/blog/lib/masonry/ImgPreview.js"></script>

<script>
    var vOssPath = 'https://sadness.oss-cn-beijing.aliyuncs.com/';
    var vOssProcess = '?x-oss-process=image/resize,m_lfit,w_348';
    var vPhotos = [
        'images/photo-Harbin/554A8512.JPG',
        'images/photo-Harbin/554A8524.JPG',
        'images/photo-Harbin/554A8531.JPG',
        'images/photo-Harbin/554A8536.JPG',
        'images/photo-Harbin/554A8555.JPG',
        'images/photo-Harbin/554A8561.JPG',
        'images/photo-Harbin/554A8596.JPG',
        'images/photo-Harbin/554A8604.jpg',
        'images/photo-Harbin/554A8614.JPG',
        'images/photo-Harbin/554A8623.JPG',
        'images/photo-Harbin/554A8644.JPG',
        'images/photo-Harbin/554A8658.JPG',
        'images/photo-Harbin/554A8662.JPG',
        'images/photo-Harbin/554A8675.JPG',
        'images/photo-Harbin/554A8696.JPG',
        'images/photo-Harbin/554A8704.JPG',
        'images/photo-Harbin/554A8718.JPG',
        'images/photo-Harbin/554A8751.JPG',
        'images/photo-Harbin/554A8796.JPG',
        'images/photo-Harbin/554A8812.jpg',
        'images/photo-Harbin/554A8824.JPG',
        'images/photo-Harbin/554A8826.JPG',
        'images/photo-Harbin/554A8841.JPG'
    ];
    vPhotos.forEach(element => {
        $("#grid").append('<li><img class="photo" src="' + vOssPath + element + vOssProcess + '" alt="' + vOssPath + element + '" style="cursor: pointer;"></li>');
    });

    new AnimOnScroll(document.getElementById('grid'), {
        minDuration : 0.4,
        maxDuration : 0.7,
        viewportFactor : 0.2
    });
    
    $(function(){  
        $(".photo").click(function(){  
            imgShow("#outerdiv", "#innerdiv", "#bigimg", $(this));
        });  
    });  
</script>