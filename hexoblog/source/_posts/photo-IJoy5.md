---
title: IJoy 5th
date: 2021-10-03 21:42:00
tags: [photo,ijoy]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-IJoy5/ijoy5logo.jpg"/>

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
        'images/photo-IJoy5/554A0930.jpg',
        'images/photo-IJoy5/554A0938.jpg',
        'images/photo-IJoy5/554A0950.jpg',
        'images/photo-IJoy5/554A0954.jpg',
        'images/photo-IJoy5/554A0958.jpg',
        'images/photo-IJoy5/554A0969.jpg',
        'images/photo-IJoy5/554A0998.jpg',
        'images/photo-IJoy5/554A1025.jpg',
        'images/photo-IJoy5/554A1049.jpg',
        'images/photo-IJoy5/554A1055.jpg',
        'images/photo-IJoy5/554A1061.jpg',
        'images/photo-IJoy5/554A1076.jpg',
        'images/photo-IJoy5/554A1080.jpg',
        'images/photo-IJoy5/554A1092.jpg',
        'images/photo-IJoy5/554A1103.jpg',
        'images/photo-IJoy5/554A1110.jpg',
        'images/photo-IJoy5/554A1173.jpg',
        'images/photo-IJoy5/554A1184.jpg',
        'images/photo-IJoy5/554A1301.jpg',
        'images/photo-IJoy5/554A1366.jpg',
        'images/photo-IJoy5/554A1379.jpg',
        'images/photo-IJoy5/554A1387.jpg',
        'images/photo-IJoy5/554A1393.jpg',
        'images/photo-IJoy5/554A1421.jpg',
        'images/photo-IJoy5/554A1424.jpg',
        'images/photo-IJoy5/554A1444.jpg',
        'images/photo-IJoy5/554A1491.jpg',
        'images/photo-IJoy5/554A1505.jpg',
        'images/photo-IJoy5/554A1510.jpg',
        'images/photo-IJoy5/554A1528.jpg',
        'images/photo-IJoy5/554A1535.jpg',
        'images/photo-IJoy5/554A1552.jpg',
        'images/photo-IJoy5/554A1557.jpg',
        'images/photo-IJoy5/554A1578.jpg',
        'images/photo-IJoy5/554A1585.jpg',
        'images/photo-IJoy5/554A1591.jpg',
        'images/photo-IJoy5/554A1595.jpg',
        'images/photo-IJoy5/554A1615.jpg',
        'images/photo-IJoy5/554A1625.jpg',
        'images/photo-IJoy5/554A1633.jpg',
        'images/photo-IJoy5/554A1637.jpg',
        'images/photo-IJoy5/554A1642.jpg',
        'images/photo-IJoy5/554A1650.jpg',
        'images/photo-IJoy5/554A1657.jpg',
        'images/photo-IJoy5/554A1658.jpg',
        'images/photo-IJoy5/554A1663.jpg',
        'images/photo-IJoy5/554A1670.jpg',
        'images/photo-IJoy5/554A1681.jpg',
        'images/photo-IJoy5/554A1708.jpg',
        'images/photo-IJoy5/554A1749.jpg',
        'images/photo-IJoy5/554A1754.jpg',
        'images/photo-IJoy5/554A1767.jpg',
        'images/photo-IJoy5/554A1773.jpg',
        'images/photo-IJoy5/554A1782.jpg'
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