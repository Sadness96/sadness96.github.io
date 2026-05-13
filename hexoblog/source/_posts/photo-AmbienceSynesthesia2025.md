---
title: 音律联觉2025：熠曲丰碑
date: 2025-05-03 16:21:00
tags: [photo,ambiencesynesthesia]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-AmbienceSynesthesia2025/ambiencesynesthesia2025logo.jpg"/>

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
        'images/photo-AmbienceSynesthesia2025/554A2796.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2798.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2811.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2816.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2822.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2835.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2842.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2856.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2863.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2874.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2885.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2894.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2902.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2919.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2928.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2941.jpg',
        'images/photo-AmbienceSynesthesia2025/554A2948.jpg',
        'images/photo-AmbienceSynesthesia2025/20250502_184503.jpg',
        'images/photo-AmbienceSynesthesia2025/20250502_184555.jpg',
        'images/photo-AmbienceSynesthesia2025/20250502_211318.jpg',
        'images/photo-AmbienceSynesthesia2025/20250502_214631.jpg'
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