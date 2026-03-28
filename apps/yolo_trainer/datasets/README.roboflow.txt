
findcarton_pallet - v5 V3_2026_Image_QC_TEST
==============================

This dataset was exported via roboflow.com on March 28, 2026 at 8:26 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 446 images.
Factory are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit within)

The following augmentation was applied to create 5 versions of each source image:
* 50% probability of horizontal flip
* Random rotation of between -6 and +6 degrees
* Random brigthness adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 0.5 pixels
* Salt and pepper noise was applied to 0.1 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* 50% probability of horizontal flip
* Random rotation of between -6 and +6 degrees
* Random brigthness adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 0.5 pixels


