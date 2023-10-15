---
title: ICOS SP漫展x五只猫
date: 2023-09-11 02:40:00
tags: [photo,icos]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-ICOSSP/icossplogo.jpg"/>

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
        'images/photo-ICOSSP/554A8010.jpg',
        'images/photo-ICOSSP/554A8026.jpg',
        'images/photo-ICOSSP/554A8027.JPG',
        'images/photo-ICOSSP/554A8036.JPG',
		'images/photo-ICOSSP/554A8043.jpg',
        'images/photo-ICOSSP/554A8061.jpg',
		'images/photo-ICOSSP/554A8065.JPG',
        'images/photo-ICOSSP/554A8066.JPG',
		'images/photo-ICOSSP/554A8069.JPG',
        'images/photo-ICOSSP/554A8075.jpg',
		'images/photo-ICOSSP/554A8077.jpg',
        'images/photo-ICOSSP/554A8087.JPG',
		'images/photo-ICOSSP/554A8099.jpg',
        'images/photo-ICOSSP/554A8106.JPG',
		'images/photo-ICOSSP/554A8108.jpg',
        'images/photo-ICOSSP/554A8115.JPG',
		'images/photo-ICOSSP/554A8119.JPG',
        'images/photo-ICOSSP/554A8128.JPG',
		'images/photo-ICOSSP/554A8129.JPG',
        'images/photo-ICOSSP/554A8132.jpg',
		'images/photo-ICOSSP/554A8140.JPG',
        'images/photo-ICOSSP/554A8144.JPG',
		'images/photo-ICOSSP/554A8146.jpg',
        'images/photo-ICOSSP/554A8149.jpg',
		'images/photo-ICOSSP/554A8153.jpg',
        'images/photo-ICOSSP/554A8161.jpg',
		'images/photo-ICOSSP/554A8165.jpg',
        'images/photo-ICOSSP/554A8167.jpg',
		'images/photo-ICOSSP/554A8177.JPG',
        'images/photo-ICOSSP/554A8184.JPG',
		'images/photo-ICOSSP/554A8190.JPG',
        'images/photo-ICOSSP/554A8193.jpg',
		'images/photo-ICOSSP/554A8198.jpg',
        'images/photo-ICOSSP/554A8210.JPG',
		'images/photo-ICOSSP/554A8216.jpg',
        'images/photo-ICOSSP/554A8219.jpg',
		'images/photo-ICOSSP/554A8230.jpg',
        'images/photo-ICOSSP/554A8232.jpg',
		'images/photo-ICOSSP/554A8237.jpg',
        'images/photo-ICOSSP/554A8244.jpg',
		'images/photo-ICOSSP/554A8248.jpg',
        'images/photo-ICOSSP/554A8260.jpg',
		'images/photo-ICOSSP/554A8263.jpg',
        'images/photo-ICOSSP/554A8268.jpg',
		'images/photo-ICOSSP/554A8271.jpg',
        'images/photo-ICOSSP/554A8274.JPG',
		'images/photo-ICOSSP/554A8275.jpg',
        'images/photo-ICOSSP/554A8299.jpg',
		'images/photo-ICOSSP/554A8304.jpg',
        'images/photo-ICOSSP/554A8306.jpg'
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