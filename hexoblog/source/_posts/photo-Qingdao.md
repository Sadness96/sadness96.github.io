---
title: 山东 - 青岛
date: 2022-10-07 22:34:20
tags: [photo]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Qingdao/qingdaologo.jpg"/>

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
        'images/photo-Qingdao/554A5824.JPG',
        'images/photo-Qingdao/554A5833.JPG',
        'images/photo-Qingdao/554A5835.JPG',
        'images/photo-Qingdao/554A5836.JPG',
        'images/photo-Qingdao/554A5837.JPG',
        'images/photo-Qingdao/554A5841.JPG',
        'images/photo-Qingdao/554A5842.JPG',
        'images/photo-Qingdao/554A5844.JPG',
        'images/photo-Qingdao/554A5845.JPG',
        'images/photo-Qingdao/554A5848.JPG',
        'images/photo-Qingdao/554A5849.JPG',
        'images/photo-Qingdao/554A5853.JPG',
        'images/photo-Qingdao/554A5854.JPG',
        'images/photo-Qingdao/554A5855.JPG',
        'images/photo-Qingdao/554A5858.JPG',
        'images/photo-Qingdao/554A5861.JPG',
        'images/photo-Qingdao/554A5863.JPG',
        'images/photo-Qingdao/554A5867.JPG',
        'images/photo-Qingdao/554A5868.JPG',
        'images/photo-Qingdao/554A5872.JPG',
        'images/photo-Qingdao/554A5875.JPG',
        'images/photo-Qingdao/554A5886.JPG',
        'images/photo-Qingdao/554A5888.JPG',
        'images/photo-Qingdao/554A5890.JPG',
        'images/photo-Qingdao/554A5908.JPG',
        'images/photo-Qingdao/554A5914.JPG',
        'images/photo-Qingdao/554A5916.JPG',
        'images/photo-Qingdao/554A5918.JPG',
        'images/photo-Qingdao/554A5920.JPG',
        'images/photo-Qingdao/554A5921.JPG',
        'images/photo-Qingdao/554A5923.JPG',
        'images/photo-Qingdao/554A5932.JPG',
        'images/photo-Qingdao/554A5933.JPG',
        'images/photo-Qingdao/554A5934.JPG',
        'images/photo-Qingdao/554A5940.JPG',
        'images/photo-Qingdao/554A5944.JPG',
        'images/photo-Qingdao/554A5947.JPG',
        'images/photo-Qingdao/554A5956.JPG',
        'images/photo-Qingdao/554A5963.JPG',
        'images/photo-Qingdao/554A5982.JPG',
        'images/photo-Qingdao/554A6000.JPG',
        'images/photo-Qingdao/554A6001.JPG',
        'images/photo-Qingdao/554A6004.JPG',
        'images/photo-Qingdao/554A6006.JPG',
        'images/photo-Qingdao/554A6007.JPG',
        'images/photo-Qingdao/554A6008.JPG',
        'images/photo-Qingdao/554A6012.JPG',
        'images/photo-Qingdao/554A6016.JPG',
        'images/photo-Qingdao/554A6017.JPG',
        'images/photo-Qingdao/554A6022.JPG',
        'images/photo-Qingdao/554A6028.JPG'
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