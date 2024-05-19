---
title: 上海
date: 2024-05-05 12:06:00
tags: [photo]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Shanghai/shanghailogo.jpg"/>

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
        'images/photo-Shanghai/554A1706.JPG',
        'images/photo-Shanghai/554A1712.JPG',
        'images/photo-Shanghai/554A1719.JPG',
        'images/photo-Shanghai/554A1721.JPG',
        'images/photo-Shanghai/554A1739.JPG',
        'images/photo-Shanghai/554A1741.JPG',
        'images/photo-Shanghai/554A1742.JPG',
        'images/photo-Shanghai/554A1754.JPG',
        'images/photo-Shanghai/554A1758.JPG',
        'images/photo-Shanghai/554A1760.JPG',
        'images/photo-Shanghai/554A1763.JPG',
        'images/photo-Shanghai/554A1765.JPG',
        'images/photo-Shanghai/554A1776.JPG',
        'images/photo-Shanghai/554A1780.JPG',
        'images/photo-Shanghai/554A1790.JPG',
        'images/photo-Shanghai/554A1792.JPG',
        'images/photo-Shanghai/554A1795.JPG',
        'images/photo-Shanghai/554A1797.JPG'
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