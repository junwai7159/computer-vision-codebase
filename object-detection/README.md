# Object Detection

## Workflow (Few-shot)
1. Creating ground-truth data that contains labels of the bounding box and class corresponding to various objects present in the image
2. Coming up with mechanisms that scan through the image to identify regions (region proposals) that are likely to contain objects
3. Creating the target class variable by using the IoU metric
4. Creating the target bounding-box offset variable to make corrections to the location of the region proposal in step 2
5. Building a model that can predict the class of object along with the target bounding-box offset corresponding to the region proposal
6. Measuring the accuracy of object detection using mean average precision (mAP)

## Concepts
### Felzenszwalb's algorithm
*Efficient Graph-Based Image Segmentation*
- segmentation based on the color, texture, size, and shape compatibility of content within an
image

### Selective Search
*Selective Search for Object Recognition*
- region proposal algorithm
- generate proposals of regions that are likely to be grouped together based on their pixel intensities
- group pixels based on the hierarchical grouping of similar pixels (leverages the color, texture, size, and shape compatibility of content within an image)
- Workflow:
  1. over-segments an image by grouping pixels based on the preceding attributes
  2. iterates through these over-segmented groups and groups them based on similarity
    - at each iteration, it combines smaller regions to form a larger region
