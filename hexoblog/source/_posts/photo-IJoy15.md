---
title: IJoy 15th
date: 2024-04-05 11:40:00
tags: [photo,ijoy]
categories: Photo
---
<img src="https://sadness96.github.io/images/blog/photo-IJoy15/ijoy15logo.jpg"/>

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
        'images/photo-IJoy15/554A0003.jpg',
        'images/photo-IJoy15/554A0014.jpg',
        'images/photo-IJoy15/554A0019.jpg',
        'images/photo-IJoy15/554A0027.jpg',
        'images/photo-IJoy15/554A0031.jpg',
        'images/photo-IJoy15/554A0032.jpg',
        'images/photo-IJoy15/554A0034.jpg',
        'images/photo-IJoy15/554A0052.jpg',
        'images/photo-IJoy15/554A0056.jpg',
        'images/photo-IJoy15/554A0058.jpg',
        'images/photo-IJoy15/554A0074.jpg',
        'images/photo-IJoy15/554A0086.jpg',
        'images/photo-IJoy15/554A0100.jpg',
        'images/photo-IJoy15/554A0132.jpg',
        'images/photo-IJoy15/554A0137.jpg',
        'images/photo-IJoy15/554A0151.jpg',
        'images/photo-IJoy15/554A0155.jpg',
        'images/photo-IJoy15/554A0161.jpg',
        'images/photo-IJoy15/554A0198.jpg',
        'images/photo-IJoy15/554A0205.jpg',
        'images/photo-IJoy15/554A0209.jpg',
        'images/photo-IJoy15/554A0212.jpg',
        'images/photo-IJoy15/554A0218.jpg',
        'images/photo-IJoy15/554A0225.jpg',
        'images/photo-IJoy15/554A0234.jpg',
        'images/photo-IJoy15/554A0242.jpg',
        'images/photo-IJoy15/554A0247.jpg',
        'images/photo-IJoy15/554A0248.jpg',
        'images/photo-IJoy15/554A0259.jpg',
        'images/photo-IJoy15/554A9463.jpg',
        'images/photo-IJoy15/554A9478.jpg',
        'images/photo-IJoy15/554A9491.jpg',
        'images/photo-IJoy15/554A9504.jpg',
        'images/photo-IJoy15/554A9514.jpg',
        'images/photo-IJoy15/554A9531.jpg',
        'images/photo-IJoy15/554A9533.jpg',
        'images/photo-IJoy15/554A9568.jpg',
        'images/photo-IJoy15/554A9571.jpg',
        'images/photo-IJoy15/554A9617.jpg',
        'images/photo-IJoy15/554A9619.jpg',
        'images/photo-IJoy15/554A9643.jpg',
        'images/photo-IJoy15/554A9646.jpg',
        'images/photo-IJoy15/554A9656.jpg',
        'images/photo-IJoy15/554A9660.jpg',
        'images/photo-IJoy15/554A9681.jpg',
        'images/photo-IJoy15/554A9692.jpg',
        'images/photo-IJoy15/554A9695.jpg',
        'images/photo-IJoy15/554A9705.jpg',
        'images/photo-IJoy15/554A9714.jpg',
        'images/photo-IJoy15/554A9771.jpg',
        'images/photo-IJoy15/554A9776.jpg',
        'images/photo-IJoy15/554A9782.jpg',
        'images/photo-IJoy15/554A9793.jpg',
        'images/photo-IJoy15/554A9803.jpg',
        'images/photo-IJoy15/554A9812.jpg',
        'images/photo-IJoy15/554A9825.jpg',
        'images/photo-IJoy15/554A9837.jpg',
        'images/photo-IJoy15/554A9841.jpg',
        'images/photo-IJoy15/554A9851.jpg',
        'images/photo-IJoy15/554A9855.jpg',
        'images/photo-IJoy15/554A9872.jpg',
        'images/photo-IJoy15/554A9879.jpg',
        'images/photo-IJoy15/554A9881.jpg',
        'images/photo-IJoy15/554A9891.jpg',
        'images/photo-IJoy15/554A9897.jpg',
        'images/photo-IJoy15/554A9913.jpg',
        'images/photo-IJoy15/554A9920.jpg',
        'images/photo-IJoy15/554A9926.jpg',
        'images/photo-IJoy15/554A9945.jpg',
        'images/photo-IJoy15/554A9950.jpg',
        'images/photo-IJoy15/554A9953.jpg',
        'images/photo-IJoy15/554A9962.jpg',
        'images/photo-IJoy15/554A9964.jpg',
        'images/photo-IJoy15/554A9966.jpg',
        'images/photo-IJoy15/554A9979.jpg',
        'images/photo-IJoy15/554A9980.jpg',
        'images/photo-IJoy15/554A9996.jpg',
        'images/photo-IJoy15/554A9998.jpg',
        'images/photo-IJoy15/554A9999.jpg'
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