# DualCNN-TF
Implementation of the DualCNN model with Tensorflow(Tensorlayer).

## Reference:
Pan J, Liu S, Sun D, et al. Learning Dual Convolutional Neural Networks for Low-Level Vision[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 3070-3079.

## result of super-resolution
<figure class="third">
    <img src="https://github.com/galad-loth/DualCNN-TF/blob/master/data/test/sr/zebra.png"  title="zebra" width="400" >
    <img src="https://github.com/galad-loth/DualCNN-TF/blob/master/test_result/sr/zebra.png_imglr.png"  title="zebra.png_imglr" width="400" >
    <img src="https://github.com/galad-loth/DualCNN-TF/blob/master/test_result/sr/zebra.png_imgsr.png"  title="zebra.png_imgsr" width="400" >
</figure>

## result of edge-preserving filtering (relative total-variation):
<figure class="half">
    <img src="https://github.com/galad-loth/DualCNN-TF/blob/master/data/test/epf/384022.jpg"  title="384022" width="400" >
    <img src="https://github.com/galad-loth/DualCNN-TF/blob/master/test_result/epf/384022.png"  title="384022_f" width="400" >
</figure>

## result of de-rain:
<figure class="half">
    <img src="https://github.com/galad-loth/DualCNN-TF/blob/master/data/test/derain/rain-009.png"  title="rain-009" width="400" >
    <img src="https://github.com/galad-loth/DualCNN-TF/blob/master/test_result/derain/rain-009.png_derain.png"  title="rain-009_derain" width="400" >
</figure>



