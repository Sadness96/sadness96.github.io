---
title: IJoy 18th
date: 2024-08-18 10:30:00
tags: [photo,ijoy]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-IJoy18/ijoy18logo.jpg"/>

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
        'images/photo-IJoy18/554A1864.jpg',
        'images/photo-IJoy18/554A1873.jpg',
        'images/photo-IJoy18/554A1881.jpg',
        'images/photo-IJoy18/554A1904.jpg',
        'images/photo-IJoy18/554A1924.jpg',
        'images/photo-IJoy18/554A1930.jpg',
        'images/photo-IJoy18/554A1957.jpg',
        'images/photo-IJoy18/554A1959.jpg',
        'images/photo-IJoy18/554A1982.jpg',
        'images/photo-IJoy18/554A1988.jpg',
        'images/photo-IJoy18/554A1999.jpg',
        'images/photo-IJoy18/554A2032.jpg',
        'images/photo-IJoy18/554A2038.jpg',
        'images/photo-IJoy18/554A2044.jpg',
        'images/photo-IJoy18/554A2069.jpg',
        'images/photo-IJoy18/554A2077.jpg',
        'images/photo-IJoy18/554A2082.jpg',
        'images/photo-IJoy18/554A2090.jpg',
        'images/photo-IJoy18/554A2099.jpg',
        'images/photo-IJoy18/554A2112.jpg',
        'images/photo-IJoy18/554A2120.jpg',
        'images/photo-IJoy18/554A2139.jpg',
        'images/photo-IJoy18/554A2144.jpg',
        'images/photo-IJoy18/554A2153.jpg',
        'images/photo-IJoy18/554A2155.jpg',
        'images/photo-IJoy18/554A2169.jpg',
        'images/photo-IJoy18/554A2178.jpg',
        'images/photo-IJoy18/554A2183.jpg',
        'images/photo-IJoy18/554A2198.jpg',
        'images/photo-IJoy18/554A2207.jpg',
        'images/photo-IJoy18/554A2233.jpg',
        'images/photo-IJoy18/554A2271.jpg',
        'images/photo-IJoy18/554A2272.jpg',
        'images/photo-IJoy18/554A2291.jpg',
        'images/photo-IJoy18/554A2296.jpg',
        'images/photo-IJoy18/554A2300.jpg',
        'images/photo-IJoy18/554A2303.jpg',
        'images/photo-IJoy18/554A2307.jpg',
        'images/photo-IJoy18/554A2337.jpg',
        'images/photo-IJoy18/554A2342.jpg',
        'images/photo-IJoy18/554A2352.jpg',
        'images/photo-IJoy18/554A2361.jpg',
        'images/photo-IJoy18/554A2367.jpg',
        'images/photo-IJoy18/554A2376.jpg',
        'images/photo-IJoy18/554A2383.jpg',
        'images/photo-IJoy18/554A2395.jpg'
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