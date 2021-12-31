---
title: IDO 35th
date: 2021-07-18 14:24:00
tags: [photo,ido]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Ido35/ido35logo.jpg"/>

<!-- more -->
<ul class="grid effect-1" id="grid">

</ul>

<link rel="stylesheet" type="text/css" href="/blog/lib/masonry/default.css" />
<link rel="stylesheet" type="text/css" href="/blog/lib/masonry/component.css" />
<script src="https://cdn.bootcss.com/jquery/3.4.1/jquery.min.js"></script>
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
        'images/photo-Ido35/554A0541.jpg',
        'images/photo-Ido35/554A0557.jpg',
        'images/photo-Ido35/554A0565.jpg',
        'images/photo-Ido35/554A0570.jpg',
        'images/photo-Ido35/554A0585.jpg',
        'images/photo-Ido35/554A0590.jpg',
        'images/photo-Ido35/554A0612.jpg',
        'images/photo-Ido35/554A0645.jpg',
        'images/photo-Ido35/554A0652.jpg',
        'images/photo-Ido35/554A0665.jpg',
        'images/photo-Ido35/554A0670.jpg',
        'images/photo-Ido35/554A0672.jpg'
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