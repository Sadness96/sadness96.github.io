---
title: IDO 36th
date: 2021-10-02 21:39:00
tags: [photo,ido]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Ido36/ido36logo.jpg"/>

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
        'images/photo-Ido36/554A0778.jpg',
        'images/photo-Ido36/554A0781.jpg',
        'images/photo-Ido36/554A0788.jpg',
        'images/photo-Ido36/554A0796.jpg',
        'images/photo-Ido36/554A0799.jpg',
        'images/photo-Ido36/554A0809.jpg',
        'images/photo-Ido36/554A0850.jpg',
        'images/photo-Ido36/554A0853.jpg',
        'images/photo-Ido36/554A0860.jpg',
        'images/photo-Ido36/554A0862.jpg',
        'images/photo-Ido36/554A0868.jpg',
        'images/photo-Ido36/554A0889.jpg',
        'images/photo-Ido36/554A0897.jpg'
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