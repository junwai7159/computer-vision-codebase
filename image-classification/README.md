# Image Classification

## Concepts
### Class Activation Map (CAM)
*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*
![grad-cam](./media/grad-cam.png)

- If a certain pixel is important, then the CNN will have a large activation at those pixels
- If a certain convolutional channel is important with respect to the required class, the gradients at that channel will be very large
