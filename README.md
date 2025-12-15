<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-Sichuan-Landslide (2025/12/16)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>Sichuan-Landslide</b> Singleclass based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512 pixels tiled PNG 
<a href="https://drive.google.com/file/d/1nDFiiJjUerxeEPY2d4n4h5tH6FN6oiKu/view?usp=sharing">
<b>Augmented-Tiled-Sichuan-Landslide-ImageMask-Dataset.zip</b></a>
which was derived by us from <br><br>
landslide.zip  in 
<a href="https://www.scidb.cn/en/detail?dataSetId=803952485596135424">
High-precision aerial imagery and interpretation dataset of landslide and <br>
debris flow disaster in Sichuan and surrounding areas
</a>
<br><br>
<b>Divide-and-Conquer Strategy</b><br>
The pixel size of images and masks in the <b>Sichuan-Landslide</b> dataset is 1181x1181 pixels, which is slightly 
large to use for our segmentation model.<br>
Therefore, we first generated an <b>Resized dataset</b> of 1024x1024 pixels, which are multiple of 512 respectively, 
, and then generated our Augmented-Tiled-Sichuan-Landslide dataset from the <b>Resized one</b>.
<br><br>
Please see also our experiment 
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Drone-Image">
TensorFlow-FlexUNet-Tiled-Image-Segmentation-Drone-Image</a>
<br>
<br>
<b>1. Tiled Image Mask Dataset</b><br>
We generated a 512 x 512 pixels tiledly-split dataset from the 1024 x 1024 pixels <b>Resized dastaset</b> 
by using our offline augmentation tool <a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator</a>
<br>
<br>
<b>2. Train Model by Tiled ImageMask Dataset</b><br>
We trained and validated the TensorFlowFlexUNet model for the Sichuan-Landslide by using the 
Tiled-Sichuan-Landslide dataset.
<br><br>
<b>3. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict mask regions for the mini_test images 
with the resolution of resized 1024 x 1024 pixels.
<br><br>
<hr>
<b>Actual Tiled Image Segmentation for the Sichuan-Landslide of 1024x1024 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/images/Landslide001.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/masks/LandslideLabel001.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test_output_tiled/Landslide001.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/images/Landslide010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/masks/LandslideLabel010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test_output_tiled/Landslide010.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/images/Landslide032.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/masks/LandslideLabel032.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test_output_tiled/Landslide032.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The dataset used here was derived from<br><br>
landslide.zip  in 
<a href="https://www.scidb.cn/en/detail?dataSetId=803952485596135424">
High-precision aerial imagery and interpretation dataset of landslide and <br>
debris flow disaster in Sichuan and surrounding areas
</a>
<br><br>
<b>Description</b><br>
Based on the landslides and debris flow disasters high-resolution digital orthophoto map (with the resolution of 0.2 m~0.9 m) in Sichuan province and its surrounding area since 2008, this paper uses visual interpretation methods to extract landslides and debris flow disasters and label the disasters. A set of the most accurate and most typical aerial imagery and interpretation data sets of landslides and debris flows are developed. The data set contains 107 typical landslide and debris flow hazard images, annotation data and description files, involving four types of disasters, including earthquake landslides, rainfall landslides, gully debris flows and slope debris flows. The disasters selected for the data sets covers the “5·12” Wenchuan, “4·20” Lushan, “8·8” Jiuzhaigou earthquake affected area, as well as the areas along the Jinsha River and Dadu River. The quality of this data set is very good, because of high-precision image data sources, interpretation and annotations by geohazard experts, and detailed disaster information. Compared with previous relevant data sets, this data set is more valuable in terms of data source quality, data set completeness and potential applications. It can not only be used for automatic interpretation of landslides and debris flow disasters, but also for disaster distribution and risk assessment research.
<br><br>
<b>Citation</b><br>
Zeng Chao, Cao Zhenyu, Su Fenghuan, et al. High-precision aerial imagery and interpretation dataset of landslide 
<br>
and debris flow disaster in Sichuan and surrounding areas[DS/OL]. V1. 
<br>Science Data Bank, 2021[2025-12-15]. https://doi.org/10.11922/sciencedb.j00001.00222. 
<br>DOI:10.11922/sciencedb.j00001.00222.
<br><br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/">
CC BY 4.0
</a>
<br>
<br>
<h3>
2  Sichuan-Landslide ImageMask Dataset
</h3>
<h4>2.1 Download  Sichuan-Landslide</h4>
 If you would like to train this Sichuan-Landslide Segmentation model by yourself,
 please download <a href="https://drive.google.com/file/d/1nDFiiJjUerxeEPY2d4n4h5tH6FN6oiKu/view?usp=sharing">
 <b>Augmented-Tiled-Sichuan-Landslide-ImageMask-Dataset.zip</b></a>
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─Sichuan-Landslide
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Sichuan-Landslide Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Sichuan-Landslide/Sichuan-Landslide_Statistics.png" width="512" height="auto"><br>
<br>

As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br> 

<h4>2.2 Sichuan-Landslide Derivation</h4>
The original landslide dataset contains 1181x1181 pixels 59 TIF images and their corresponding TIF labels in a single landslide folder.  

<pre>
./landslide
├─Landslide001.tif
...
├─Landslide059.tif

├─LandslideLabel001.tif
...
└─LandslideLabel059.tif
</pre>
We used the following two Python scripts to generate our 512x512 pixels tiled dataset.
<ul>
<li><a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
</ul>


<h4>2.3 Sichuan-Landslide Samples</h4>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Sichuan-Landslide TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Sichuan-Landslide/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Sichuan-Landslide and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and a large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
num_classes    = 2

base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;           
; Landslide 1+1 classes
;                 Landslide: mazenda
rgb_map={(0,0,0):0,(255,0,255):1
</pre>

<b>Epoch change tiled inference callback</b><br>
Enabled <a href="./src/EpochChangeTileInferencer.py">epoch_change_tiled_infer callback (EpochChangeTiledInferencer.py)</a></b>.<br>
<pre>
[train]
poch_change_infer     = False
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_infer     = True
epoch_change_infer_dir =  "./epoch_change_infer"
</pre>

By using this callback, on every epoch_change, the tiled inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3,4)</b><br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 32,33,34,35)</b><br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 65,66,67,68)</b><br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 68 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/asset/train_console_output_at_epoch68.png" width="880" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Sichuan-Landslide/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Sichuan-Landslide/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Sichuan-Landslide</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Sichuan-Landslide.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/asset/evaluate_console_output_at_epoch68.png" width="880" height="auto">
<br><br>Image-Segmentation-Aerial-Imagery

<a href="./projects/TensorFlowFlexUNet/Sichuan-Landslide/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this Sichuan-Landslide/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.2103
dice_coef_multiclass,0.9104
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Sichuan-Landslide</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Sichuan-Landslide.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>_inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for the Sichuan-Landslide 1024x1024 pixels</b><br>
As shown below, the tiled inferred masks predicted by our segmentation model trained on the 
 dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<br><br>
<table>
<tr>

<th>Input: Image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/images/Landslide003.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/masks/LandslideLabel003.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test_output_tiled/Landslide003.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/images/Landslide010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/masks/LandslideLabel010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test_output_tiled/Landslide010.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/images/Landslide014.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/masks/LandslideLabel014.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test_output_tiled/Landslide014.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/images/Landslide028.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/masks/LandslideLabel028.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test_output_tiled/Landslide028.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/images/Landslide031.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/masks/LandslideLabel031.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test_output_tiled/Landslide031.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/images/Landslide036.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test/masks/LandslideLabel036.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Sichuan-Landslide/mini_test_output_tiled/Landslide036.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Landslide4Sense: Multi-sensor landslide detection competition & benchmark dataset</b><br>
Institute of Advanced Research in Artificial Intelligence<br>
<a href="https://github.com/iarai/Landslide4Sense-2022">
https://github.com/iarai/Landslide4Sense-2022</a>
<br>
<br>
<b>2. The Outcome of the 2022 Landslide4Sense Competition:<br> Advanced Landslide Detection From Multisource Satellite Imagery
</b><br>
Omid Ghorbanzadeh; Yonghao Xu; Hengwei Zhao; Junjue Wang; Yanfei Zhong; Dong Zhao<br>
<a href="https://ieeexplore.ieee.org/document/9944085">
https://ieeexplore.ieee.org/document/9944085
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense
</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide
</a>

