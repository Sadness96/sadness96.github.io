---
title: 喘气动漫
date: 2024-08-26 10:38:00
tags: [photo,chuanqi]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-ChuanQiDongMan/chuanqilogo.jpg"/>

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

<script>
    var vOssPath = 'https://sadness.oss-cn-beijing.aliyuncs.com/';
    var vOssProcess = '?x-oss-process=image/resize,m_lfit,w_348';
    var vPhotos = [
        'images/photo-ChuanQiDongMan/554A2404.jpg',
        'images/photo-ChuanQiDongMan/554A2424.jpg',
        'images/photo-ChuanQiDongMan/554A2428.jpg',
        'images/photo-ChuanQiDongMan/554A2436.jpg',
        'images/photo-ChuanQiDongMan/554A2441.jpg',
        'images/photo-ChuanQiDongMan/554A2449.jpg',
        'images/photo-ChuanQiDongMan/554A2450.jpg',
        'images/photo-ChuanQiDongMan/554A2463.jpg',
        'images/photo-ChuanQiDongMan/554A2466.jpg',
        'images/photo-ChuanQiDongMan/554A2489.jpg',
        'images/photo-ChuanQiDongMan/554A2500.jpg',
        'images/photo-ChuanQiDongMan/554A2508.jpg'
    ];
    vPhotos.forEach(element => {
        $("#grid").append('<li><img class="photo" src="' + vOssPath + element + vOssProcess + '" data-zoom-src="' + vOssPath + element + '"></li>');
    });

    new AnimOnScroll(document.getElementById('grid'), {
        minDuration : 0.4,
        maxDuration : 0.7,
        viewportFactor : 0.2
    });

    mediumZoom('.post-body .photo', { background: 'rgba(0,0,0,0.7)' });
</script>