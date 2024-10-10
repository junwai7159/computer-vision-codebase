# Image Classification

## Concepts
### Class Activation Map (CAM)
*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*
![grad-cam](./media/grad-cam.png)

- If a certain pixel is important, then the CNN will have a large activation at those pixels
- If a certain convolutional channel is important with respect to the required class, the gradients at that channel will be very large

### Caveats
- Imbalanced data
  - confusion matrix
  - loss function (binary or categorical cross-entropy) ensures that the loss values are high when the amount of misclassification is high
  - higher class weights to rare class image
  - over-sample rare class image
  - data augmentation
  - transfer learning
- The size of the object (small) within an image
  - object detection: divide input image into smaller grid cells, then identify whether a grid cell contains the object of interest
  - model is trained and inferred on images with high resolution
- Data drift
- The number of nodes in the flatten layer
  - typically around 500-5000 nodes
- Image size
  - images of objects might not lose information if resized
  - images of text might lose considerable information
