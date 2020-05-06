# 數位視覺效果 作業2
 B05902050 黃子源
 B05902106 宋昶松
## 介紹
在這個作業中，我們利用image warping, Harris corner detector, feature matching, image matching, blending來實作image stitching, 將數張360度環繞的照片變成一張全景圖。
## Warping
我們利用課程中教的cylindrical warping來將每一張照片投影到全景圖相對應的圓柱體上。其中warping的轉換公式為：
$$x' = s*tan^{-1}(\frac{x}{f})$$
$$y' = s\frac{y}{\sqrt{x^2 + f^2}}$$
其中$x'$,$y'$為轉換後對應到的位置，為了讓整體圖片的變形不要太誇張，我取$s = f$。因為AutoStich在我們windows 32位元虛擬機中無法執行，因此我們先利用開源exif分析網頁找出以mm為單位的焦距，再利用$F(pixels) = F(mm) * 5472 / 13.2$來轉換成pixel焦距。

利用上式轉換，由於這個warping轉換並不是一對一，會有某些輸出點沒有對應點，因此這些輸出點都會是0
## Harris Corner Detector
## Feature Matching
## Image Matching
## Blending
## 參考
- online meta data viewer: http://metapicz.com/#landing
- multi-band blending: https://github.com/cynricfu/multi-band-blending/blob/master/multi_band_blending.py
