---
title: 中国 - 北京
date: 2021-03-04 23:42:00
tags: [photo]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Beijing/beijinglogo.jpg"/>

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
        'images/photo-Beijing/554A0271.JPG',
        'images/photo-Beijing/554A0358.JPG',
        'images/photo-Beijing/554A0361.JPG',
        'images/photo-Beijing/554A0364.JPG',
        'images/photo-Beijing/554A0383.jpg',
        'images/photo-Beijing/554A0394.JPG',
        'images/photo-Beijing/554A0414.JPG',
        'images/photo-Beijing/554A0456.JPG',
        'images/photo-Beijing/554A0459.JPG',
        'images/photo-Beijing/554A0466.JPG',
        'images/photo-Beijing/554A0473.JPG',
        'images/photo-Beijing/554A0491.JPG',
        'images/photo-Beijing/554A0493.JPG',
        'images/photo-Beijing/554A0496.JPG',
        'images/photo-Beijing/554A0499.JPG',
        'images/photo-Beijing/554A0500.JPG',
        'images/photo-Beijing/554A1918.JPG',
        'images/photo-Beijing/554A1925.JPG',
        'images/photo-Beijing/554A1929.JPG',
        'images/photo-Beijing/554A1931.JPG',
        'images/photo-Beijing/554A1933.JPG',
        'images/photo-Beijing/554A7122.JPG',
        'images/photo-Beijing/554A7125.JPG',
        'images/photo-Beijing/554A7128.JPG',
        'images/photo-Beijing/554A7130.JPG',
        'images/photo-Beijing/554A7133.JPG',
        'images/photo-Beijing/554A7138.JPG',
        'images/photo-Beijing/554A7140.JPG',
        'images/photo-Beijing/554A7142.JPG',
        'images/photo-Beijing/554A7149.JPG',
        'images/photo-Beijing/554A7152.JPG',
        'images/photo-Beijing/554A7154.JPG',
        'images/photo-Beijing/554A7159.JPG',
        'images/photo-Beijing/554A7165.JPG',
        'images/photo-Beijing/554A7170.JPG',
        'images/photo-Beijing/554A7174.JPG',
        'images/photo-Beijing/554A7177.JPG',
        'images/photo-Beijing/554A7183.JPG',
        'images/photo-Beijing/554A7187.JPG',
        'images/photo-Beijing/554A7192.JPG',
        'images/photo-Beijing/554A7195.JPG',
        'images/photo-Beijing/554A7200.JPG',
        'images/photo-Beijing/554A7202.JPG',
        'images/photo-Beijing/554A7203.JPG',
        'images/photo-Beijing/554A7205.JPG',
        'images/photo-Beijing/554A7214.JPG',
        'images/photo-Beijing/554A7218.JPG',
        'images/photo-Beijing/554A7220.JPG',
        'images/photo-Beijing/554A7228.JPG',
        'images/photo-Beijing/554A7232.JPG',
        'images/photo-Beijing/554A7236.JPG',
        'images/photo-Beijing/554A7240.JPG',
        'images/photo-Beijing/554A7245.JPG',
        'images/photo-Beijing/554A7246.JPG',
        'images/photo-Beijing/554A7254.JPG',
        'images/photo-Beijing/554A7255.JPG',
        'images/photo-Beijing/554A7259.JPG',
        'images/photo-Beijing/554A7266.jpg',
        'images/photo-Beijing/554A7278.JPG'
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