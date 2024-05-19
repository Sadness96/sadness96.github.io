---
title: 浙江 - 杭州
date: 2024-05-04 12:05:00
tags: [photo]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-Hangzhou/hangzhoulogo.jpg"/>

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
        'images/photo-Hangzhou/554A0601.JPG',
        'images/photo-Hangzhou/554A0610.JPG',
        'images/photo-Hangzhou/554A0621.JPG',
        'images/photo-Hangzhou/554A0623.JPG',
        'images/photo-Hangzhou/554A0626.JPG',
        'images/photo-Hangzhou/554A0632.JPG',
        'images/photo-Hangzhou/554A0639.JPG',
        'images/photo-Hangzhou/554A0643.JPG',
        'images/photo-Hangzhou/554A0653.JPG',
        'images/photo-Hangzhou/554A0654.JPG',
        'images/photo-Hangzhou/554A0659.JPG',
        'images/photo-Hangzhou/554A0666.JPG',
        'images/photo-Hangzhou/554A0669.JPG',
        'images/photo-Hangzhou/554A0678.JPG',
        'images/photo-Hangzhou/554A0684.JPG',
        'images/photo-Hangzhou/554A0686.JPG',
        'images/photo-Hangzhou/554A0687.JPG',
        'images/photo-Hangzhou/554A0689.JPG',
        'images/photo-Hangzhou/554A0695.JPG',
        'images/photo-Hangzhou/554A0699.JPG',
        'images/photo-Hangzhou/554A0705.JPG',
        'images/photo-Hangzhou/554A0716.JPG',
        'images/photo-Hangzhou/554A0717.JPG',
        'images/photo-Hangzhou/554A0722.JPG',
        'images/photo-Hangzhou/554A0725.JPG',
        'images/photo-Hangzhou/554A0728.JPG',
        'images/photo-Hangzhou/554A0733.JPG',
        'images/photo-Hangzhou/554A0735.JPG',
        'images/photo-Hangzhou/554A0737.JPG',
        'images/photo-Hangzhou/554A0741.JPG',
        'images/photo-Hangzhou/554A0742.JPG',
        'images/photo-Hangzhou/554A0745.JPG',
        'images/photo-Hangzhou/554A0749.JPG',
        'images/photo-Hangzhou/554A0750.JPG',
        'images/photo-Hangzhou/554A0752.JPG',
        'images/photo-Hangzhou/554A0756.JPG',
        'images/photo-Hangzhou/554A0763.JPG',
        'images/photo-Hangzhou/554A0778.JPG',
        'images/photo-Hangzhou/554A0780.JPG',
        'images/photo-Hangzhou/554A0786.JPG',
        'images/photo-Hangzhou/554A0790.JPG',
        'images/photo-Hangzhou/554A0791.JPG',
        'images/photo-Hangzhou/554A0792.JPG',
        'images/photo-Hangzhou/554A0794.JPG',
        'images/photo-Hangzhou/554A0795.JPG',
        'images/photo-Hangzhou/554A0797.JPG',
        'images/photo-Hangzhou/554A0798.JPG',
        'images/photo-Hangzhou/554A0799.JPG',
        'images/photo-Hangzhou/554A0801.JPG',
        'images/photo-Hangzhou/554A0803.JPG',
        'images/photo-Hangzhou/554A0805.JPG',
        'images/photo-Hangzhou/554A0806.JPG',
        'images/photo-Hangzhou/554A0808.JPG',
        'images/photo-Hangzhou/554A0809.JPG',
        'images/photo-Hangzhou/554A0816.JPG',
        'images/photo-Hangzhou/554A0817.JPG',
        'images/photo-Hangzhou/554A0819.JPG',
        'images/photo-Hangzhou/554A0827.JPG',
        'images/photo-Hangzhou/554A0829.JPG',
        'images/photo-Hangzhou/554A0833.JPG',
        'images/photo-Hangzhou/554A0843.JPG',
        'images/photo-Hangzhou/554A0844.JPG',
        'images/photo-Hangzhou/554A0850.JPG',
        'images/photo-Hangzhou/554A0853.JPG',
        'images/photo-Hangzhou/554A0854.JPG',
        'images/photo-Hangzhou/554A0860.JPG',
        'images/photo-Hangzhou/554A0861.JPG',
        'images/photo-Hangzhou/554A0863.JPG',
        'images/photo-Hangzhou/554A0866.JPG',
        'images/photo-Hangzhou/554A0870.JPG',
        'images/photo-Hangzhou/554A0877.JPG',
        'images/photo-Hangzhou/554A0881.JPG',
        'images/photo-Hangzhou/554A0882.JPG',
        'images/photo-Hangzhou/554A0887.JPG',
        'images/photo-Hangzhou/554A0899.JPG',
        'images/photo-Hangzhou/554A0908.JPG',
        'images/photo-Hangzhou/554A0910.JPG',
        'images/photo-Hangzhou/554A0920.JPG',
        'images/photo-Hangzhou/554A0921.JPG',
        'images/photo-Hangzhou/554A0925.JPG',
        'images/photo-Hangzhou/554A0926.JPG',
        'images/photo-Hangzhou/554A0927.JPG'
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