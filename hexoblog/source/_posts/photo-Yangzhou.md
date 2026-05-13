---
title: 江苏 - 扬州
date: 2025-05-02 05:12:00
tags: [photo]
categories: Photo
---
烟花<del style="color:OrangeRed;">三月</del>(四月)下扬州
<img src="https://sadness96.github.io/images/blog/photo-Yangzhou/yangzhoulogo.jpg"/>

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
        'images/photo-Yangzhou/554A2535.JPG',
        'images/photo-Yangzhou/554A2537.JPG',
        'images/photo-Yangzhou/554A2538.JPG',
        'images/photo-Yangzhou/554A2541.JPG',
        'images/photo-Yangzhou/554A2550.JPG',
        'images/photo-Yangzhou/554A2552.JPG',
        'images/photo-Yangzhou/554A2554.JPG',
        'images/photo-Yangzhou/554A2556.JPG',
        'images/photo-Yangzhou/554A2557.JPG',
        'images/photo-Yangzhou/554A2584.JPG',
        'images/photo-Yangzhou/554A2589.JPG',
        'images/photo-Yangzhou/554A2594.JPG',
        'images/photo-Yangzhou/554A2603.JPG',
        'images/photo-Yangzhou/554A2612.JPG',
        'images/photo-Yangzhou/554A2615.JPG',
        'images/photo-Yangzhou/554A2616.JPG',
        'images/photo-Yangzhou/554A2623.JPG',
        'images/photo-Yangzhou/554A2668.JPG',
        'images/photo-Yangzhou/554A2669.JPG',
        'images/photo-Yangzhou/554A2678.JPG',
        'images/photo-Yangzhou/554A2783.JPG',
        'images/photo-Yangzhou/554A2791.JPG',
        'images/photo-Yangzhou/554A2792.JPG',
        'images/photo-Yangzhou/554A2794.JPG',
        'images/photo-Yangzhou/20250430_101732.jpg',
        'images/photo-Yangzhou/20250501_142602.jpg',
        'images/photo-Yangzhou/20250501_143457.jpg',
        'images/photo-Yangzhou/20250501_144250.jpg',
        'images/photo-Yangzhou/20250501_144259.jpg',
        'images/photo-Yangzhou/20250501_144309.jpg',
        'images/photo-Yangzhou/20250501_144318.jpg',
        'images/photo-Yangzhou/20250501_144801.jpg',
        'images/photo-Yangzhou/20250501_145605.jpg',
        'images/photo-Yangzhou/20250501_151302.jpg',
        'images/photo-Yangzhou/20250501_152050.jpg',
        'images/photo-Yangzhou/20250501_152309.jpg',
        'images/photo-Yangzhou/20250501_153612.jpg',
        'images/photo-Yangzhou/20250501_154511.jpg',
        'images/photo-Yangzhou/20250501_154807.jpg'
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