import matplotlib.pyplot as plt 
import torchvision
import numpy as np

def plot_masks(X,Y,Y_P,IOU,rows=2,cols=4,im_scale=4):

  IOU = IOU.numpy()

  iou_minsort = np.argsort(IOU)

  fig, axs = plt.subplots(rows,cols,figsize=(im_scale*cols,im_scale*rows))

  to_pil = torchvision.transforms.ToPILImage()

  for ax, a in zip(axs.flatten(), iou_minsort[:rows*cols]):
    
    im = (X[a] * 255).byte()
    im = torchvision.utils.draw_segmentation_masks(im, Y[a] > 0.5, alpha=0.3, colors='red')
    im = torchvision.utils.draw_segmentation_masks(im, Y_P[a] > 0.5, alpha=0.3, colors='blue')

    ax.imshow(to_pil(im))
    ax.axis('off')
    ax.plot([], "ro", label="Ground Truth")
    ax.plot([], "bo", label="Prediction")
    ax.set_title(f"IoU={IOU[a]:.06f}")
    ax.legend()

  fig.suptitle(f'Bottom {rows*cols} IoUs')
  
  return fig

def plot_histogram(X,Y,Y_P,IOU):

  fig, (ax) = plt.subplots()
  ax.hist(IOU, bins=50)
  ax.set_title('Histogram of IoUs')

  return fig