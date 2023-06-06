---
title: 山东 - 济南
date: 2023-06-01 19:09:00
tags: [photo]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Jinan/jinanlogo.jpg"/>

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
        'images/photo-Jinan/554A7655.JPG',
        'images/photo-Jinan/554A7665.JPG',
        'images/photo-Jinan/554A7667.JPG',
        'images/photo-Jinan/554A7669.JPG',
        'images/photo-Jinan/554A7672.JPG',
        'images/photo-Jinan/554A7676.JPG',
        'images/photo-Jinan/554A7677.JPG',
        'images/photo-Jinan/554A7678.JPG',
        'images/photo-Jinan/554A7685.JPG',
        'images/photo-Jinan/554A7688.JPG',
        'images/photo-Jinan/554A7689.JPG',
        'images/photo-Jinan/554A7692.JPG',
        'images/photo-Jinan/554A7700.JPG',
        'images/photo-Jinan/554A7702.JPG',
        'images/photo-Jinan/554A7704.JPG',
        'images/photo-Jinan/554A7708.JPG',
        'images/photo-Jinan/554A7709.JPG',
        'images/photo-Jinan/554A7711.JPG',
        'images/photo-Jinan/554A7712.JPG',
        'images/photo-Jinan/554A7728.JPG',
        'images/photo-Jinan/554A7729.JPG',
        'images/photo-Jinan/554A7738.JPG',
        'images/photo-Jinan/554A7752.JPG',
        'images/photo-Jinan/554A7759.JPG',
        'images/photo-Jinan/554A7763.JPG',
        'images/photo-Jinan/554A7764.JPG',
        'images/photo-Jinan/554A7770.JPG',
        'images/photo-Jinan/554A7774.JPG',
        'images/photo-Jinan/554A7789.JPG',
        'images/photo-Jinan/554A7795.JPG',
        'images/photo-Jinan/554A7804.JPG',
        'images/photo-Jinan/554A7808.JPG',
        'images/photo-Jinan/554A7810.JPG',
        'images/photo-Jinan/554A7812.JPG',
        'images/photo-Jinan/554A7813.JPG',
        'images/photo-Jinan/554A7817.JPG',
        'images/photo-Jinan/554A7823.JPG',
        'images/photo-Jinan/554A7825.JPG',
        'images/photo-Jinan/554A7829.JPG',
        'images/photo-Jinan/554A7830.JPG',
        'images/photo-Jinan/554A7833.JPG',
        'images/photo-Jinan/554A7835.JPG',
        'images/photo-Jinan/554A7836.JPG',
        'images/photo-Jinan/554A7837.JPG',
        'images/photo-Jinan/554A7838.JPG',
        'images/photo-Jinan/554A7840.JPG',
        'images/photo-Jinan/554A7842.JPG',
        'images/photo-Jinan/554A7843.JPG',
        'images/photo-Jinan/554A7847.JPG',
        'images/photo-Jinan/554A7849.JPG',
        'images/photo-Jinan/554A7857.JPG',
        'images/photo-Jinan/554A7863.JPG',
        'images/photo-Jinan/554A7869.JPG',
        'images/photo-Jinan/554A7875.JPG',
        'images/photo-Jinan/554A7878.JPG',
        'images/photo-Jinan/554A7879.JPG',
        'images/photo-Jinan/554A7886.JPG',
        'images/photo-Jinan/554A7887.JPG',
        'images/photo-Jinan/554A7890.JPG',
        'images/photo-Jinan/554A7891.JPG',
        'images/photo-Jinan/554A7893.JPG',
        'images/photo-Jinan/554A7904.JPG',
        'images/photo-Jinan/554A7906.JPG',
        'images/photo-Jinan/554A7916.JPG',
        'images/photo-Jinan/554A7921.JPG',
        'images/photo-Jinan/554A7923.JPG',
        'images/photo-Jinan/554A7929.JPG',
        'images/photo-Jinan/554A7945.JPG',
        'images/photo-Jinan/554A7951.JPG'
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