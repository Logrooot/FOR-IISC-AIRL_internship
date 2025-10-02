# FOR-IISC-AIRL_internship

## Q1. Vision Transformer on CIFAR-10

### Analysis:

all the analysis are done for 10 epochs

### 1.Effects of image augmentations:

other hyper parameters: <br>
Batch_size=128
Epochs=10
Learning_rate=3e-4
Patch_size=4
Num_classes=10
img_size=32
Channels=3
Embed_dim=256
Num_heads=8
Depth=6
Mlp_dim=512
Drop_rate=0.1

Image Augmentations done are:

transforms.RandomCrop(32,padding=4) <br>
transforms.RandomHorizontalFlip() <br>
transforms.RandomRotation(10) <br>

Results: <br>
without augmentation:  Train Loss:0.6710, Train acc: 0.7600, Test acc: 0.6330 <br>
with augmentation   :  Train Loss:1.0664, Train acc: 0.6192, Test acc: 0.6189

After augmentation, the overall accuracy decreased compared to the non-augmented one, but the training accuracy is much lower this time which indicates the improved generalization of the model. Augmentation is a strong factor in improving accuracy in the CIFAR-10 dataset

## 2.Effects of different patch sizes: 
 
other hyper parameters: <br>
Batch_size=128
Epochs=10
Learning_rate=3e-4
Num_classes=10
img_size=32
Channels=3
Embed_dim=256
Num_heads=8
Depth=6
Mlp_dim=512
Drop_rate=0.1
optimizer used is adam
and no image augmentations are used

Results: <br>
Patch size = 4  :   Train Loss:0.6710, Train acc: 0.7600, Test acc: 0.6330 <br>
Patch size = 8  :   Train Loss:0.7290, Train acc: 0.7400, Test acc: 0.6084 <br>
Patch size = 16 :   Train Loss:0.7524, Train acc: 0.7306, Test acc: 0.5665 <br>

its observed that the overall performance of the model decreases with increase in patch size (decreases number of patches), but the training time taken reduces drastically thus efficient but due to less patches finer details are lost.

## 3.Effects of different optimizers:

other hyper parameters: <br>
Batch_size=128
Epochs=10
Learning_rate=3e-4
Patch_size=4
Num_classes=10
img_size=32
Channels=3
Embed_dim=256
Num_heads=8
Depth=6
Mlp_dim=512
Drop_rate=0.1
with no augmentations

Results: <br>

Adam    : Train Loss:0.6710, Train acc: 0.7600, Test acc: 0.6330 <br>
SGD     : Train Loss:2.2059, Train acc: 0.1853, Test acc: 0.2048 <br>
Adagrad : Train Loss:1.5042, Train acc: 0.4631, Test acc: 0.4808 <br>

Adam optimizer performed better as it uses adaptive learning rate similar to adagrad, but also utilizes momentum which makes it the better choice for ViT. 

## 4.Effects of overlapping and non overlapping patches:

other hyper parameters: <br>
Batch_size=128
Epochs=10
Learning_rate=3e-4
Patch_size=4
Num_classes=10
img_size=32
Channels=3
Embed_dim=256
Num_heads=8
Depth=6
Mlp_dim=512
Drop_rate=0.1
with no augmentations
with adam optimizer

Results: <br>
non overlapping patches           : Train Loss:0.6710, Train acc: 0.7600, Test acc: 0.6330 <br>
overlapping patches (50% overlap) : Train Loss:0.6416, Train acc: 0.7721, Test acc: 0.6472 <br>

overlapping patches take up lot more computational units due to its complexity, dosent provide much improvement in terms of model accuracy and loss (~1% improvement for 4x computation time), so non overlapping patches are widely prefered.

# Q2. Grounding Dino +SAM 2 pipeline
### Pipeline Description:
#### Input image → Grounding DINO (tiny) for text-prompted object detection → generate region seeds (points from bounding boxes) → Segment Anything 2.1 (large) for segmentation → output image with object masks.

#### 1.Load Models

* Grounding DINO (tiny) for zero-shot object detection using text prompts.

* SAM 2 (large) for image segmentation.

#### 2.Input & Preprocessing

* Load an input image.

* Provide text prompts describing target objects (e.g., “Black cat”).

* Grounding DINO detects bounding boxes for the specified objects.

#### 3.Convert Detections to Seeds

* Extract bounding box centers as region seeds.

* Format them into input_points and input_labels for SAM 2.

#### 4.Segmentation with SAM 2

* Feed seeds and image into SAM 2.

* Generate object segmentation masks.

#### 5.Visualization

* Overlay masks, bounding boxes, and seed points on the original image.

### Limitations:

* Heavy GPU/memory use (especially SAM 2.1 large).

* Results depend on the quality of text prompts and detections, and the utilized model is GroundingDino Tiny which affects the quality of reion seeds

* Limited flexibility (fixed thresholds and size).

* No error handling if detections fail.



