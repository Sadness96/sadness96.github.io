---
title: IDO 41th
date: 2023-05-01 20:55:00
tags: [photo,ido]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Ido41/ido41logo.jpg"/>

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
        'images/photo-Ido41/554A6936.jpg',
        'images/photo-Ido41/554A6948.jpg',
        'images/photo-Ido41/554A6954.jpg',
        'images/photo-Ido41/554A6956.jpg',
        'images/photo-Ido41/554A6969.jpg',
        'images/photo-Ido41/554A6975.jpg',
        'images/photo-Ido41/554A6978.jpg',
        'images/photo-Ido41/554A6988.jpg',
        'images/photo-Ido41/554A6992.jpg',
        'images/photo-Ido41/554A6997.jpg',
        'images/photo-Ido41/554A6998.jpg',
        'images/photo-Ido41/554A7008.jpg',
        'images/photo-Ido41/554A7011.jpg',
        'images/photo-Ido41/554A7015.jpg',
        'images/photo-Ido41/554A7025.jpg',
        'images/photo-Ido41/554A7027.jpg',
        'images/photo-Ido41/554A7030.jpg',
        'images/photo-Ido41/554A7037.jpg',
        'images/photo-Ido41/554A7039.jpg',
        'images/photo-Ido41/554A7049.jpg',
        'images/photo-Ido41/554A7052.jpg',
        'images/photo-Ido41/554A7057.jpg',
        'images/photo-Ido41/554A7060.jpg',
        'images/photo-Ido41/554A7066.jpg',
        'images/photo-Ido41/554A7069.jpg',
        'images/photo-Ido41/554A7077.jpg',
        'images/photo-Ido41/554A7093.jpg',
        'images/photo-Ido41/554A7102.jpg',
        'images/photo-Ido41/554A7103.jpg',
        'images/photo-Ido41/554A7111.jpg',
        'images/photo-Ido41/554A7117.jpg'
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