---
title: 新疆 - 乌鲁木齐
date: 2023-05-20 23:40:00
tags: [photo]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Urumqi/urumqilogo.jpg"/>

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
        'images/photo-Urumqi/554A7286.JPG',
        'images/photo-Urumqi/554A7294.JPG',
        'images/photo-Urumqi/554A7296.JPG',
        'images/photo-Urumqi/554A7299.JPG',
        'images/photo-Urumqi/554A7321.JPG',
        'images/photo-Urumqi/554A7328.JPG',
        'images/photo-Urumqi/554A7342.JPG',
        'images/photo-Urumqi/554A7343.JPG',
        'images/photo-Urumqi/554A7345.JPG',
        'images/photo-Urumqi/554A7359.JPG',
        'images/photo-Urumqi/554A7363.JPG',
        'images/photo-Urumqi/554A7370.JPG',
        'images/photo-Urumqi/554A7371.JPG',
        'images/photo-Urumqi/554A7405.JPG',
        'images/photo-Urumqi/554A7411.JPG',
        'images/photo-Urumqi/554A7416.JPG',
        'images/photo-Urumqi/554A7424.JPG',
        'images/photo-Urumqi/554A7425.JPG',
        'images/photo-Urumqi/554A7440.JPG',
        'images/photo-Urumqi/554A7446.JPG',
        'images/photo-Urumqi/554A7452.JPG',
        'images/photo-Urumqi/554A7458.JPG',
        'images/photo-Urumqi/554A7472.JPG',
        'images/photo-Urumqi/554A7480.JPG',
        'images/photo-Urumqi/554A7485.JPG',
        'images/photo-Urumqi/554A7487.JPG',
        'images/photo-Urumqi/554A7488.JPG',
        'images/photo-Urumqi/554A7489.JPG',
        'images/photo-Urumqi/554A7492.JPG',
        'images/photo-Urumqi/554A7494.JPG',
        'images/photo-Urumqi/554A7497.JPG',
        'images/photo-Urumqi/554A7500.JPG',
        'images/photo-Urumqi/554A7502.JPG',
        'images/photo-Urumqi/554A7505.JPG',
        'images/photo-Urumqi/554A7509.JPG',
        'images/photo-Urumqi/554A7515.JPG',
        'images/photo-Urumqi/554A7516.JPG',
        'images/photo-Urumqi/554A7517.JPG',
        'images/photo-Urumqi/554A7521.JPG',
        'images/photo-Urumqi/554A7523.JPG',
        'images/photo-Urumqi/554A7528.JPG',
        'images/photo-Urumqi/554A7542.JPG',
        'images/photo-Urumqi/554A7549.JPG',
        'images/photo-Urumqi/554A7550.JPG',
        'images/photo-Urumqi/554A7553.JPG',
        'images/photo-Urumqi/554A7555.JPG',
        'images/photo-Urumqi/554A7560.JPG'
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