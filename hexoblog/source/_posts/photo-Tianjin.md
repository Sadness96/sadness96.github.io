---
title: 天津
date: 2022-11-06 20:06:04
tags: [photo]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Tianjin/tianjinlogo.jpg"/>

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
        'images/photo-Tianjin/554A6031.JPG',
        'images/photo-Tianjin/554A6033.JPG',
        'images/photo-Tianjin/554A6035.JPG',
        'images/photo-Tianjin/554A6039.JPG',
        'images/photo-Tianjin/554A6043.JPG',
        'images/photo-Tianjin/554A6052.JPG',
        'images/photo-Tianjin/554A6056.JPG',
        'images/photo-Tianjin/554A6064.JPG',
        'images/photo-Tianjin/554A6065.JPG',
        'images/photo-Tianjin/554A6071.JPG',
        'images/photo-Tianjin/554A6075.JPG',
        'images/photo-Tianjin/554A6077.JPG',
        'images/photo-Tianjin/554A6079.JPG',
        'images/photo-Tianjin/554A6080.JPG',
        'images/photo-Tianjin/554A6082.JPG',
        'images/photo-Tianjin/554A6083.JPG',
        'images/photo-Tianjin/554A6088.JPG',
        'images/photo-Tianjin/554A6105.JPG',
        'images/photo-Tianjin/554A6106.JPG',
        'images/photo-Tianjin/554A6111.JPG',
        'images/photo-Tianjin/554A6115.JPG',
        'images/photo-Tianjin/554A6127.JPG',
        'images/photo-Tianjin/554A6142.JPG',
        'images/photo-Tianjin/554A6143.JPG',
        'images/photo-Tianjin/554A6147.JPG',
        'images/photo-Tianjin/554A6158.JPG',
        'images/photo-Tianjin/554A6189.JPG',
        'images/photo-Tianjin/20221022_180444.jpg'
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