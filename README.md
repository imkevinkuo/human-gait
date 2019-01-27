# human-gait
  
A research project on extracting features of human gait from side-view treadmill videos.  

# Feature extraction

## Pixel counting
We can determine stride cycles by simply counting the number of pixels in each frame.  
In this graph, local minima represent the frames where legs cross over each other. The frames between any 3 consecutive minima represent a complete stride cycle (two steps).  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/imgs/pixelsum.png" width="60%">  

## Vertical position and velocity
To determine vertical position of the subject, we calculate the geometric center of the head.  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/imgs/00094.png" width="20%">
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/imgs/00001red.png" width="20%">  
Vertical position vs. time:  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/imgs/10kmposition.png" width="60%">  
Once we have a graph of vertical position, we can take the 1st derivative to measure the subject's vertical velocity. The minimums in the velocity graph signify the frames where the subject's foot strikes the ground. Foot strike can also be used to determine stride cycles.  
Vertical velocity vs. time:  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/imgs/10kmvelocity.png" width="60%">  

## Ellipse of Approximation
Method documented in <a href="https://dl.acm.org/citation.cfm?id=2676612">Extracting silhouette-based characteristics for human gait analysis using one camera</a>.  
If we plot the angle of inclination (obtained by using an ellipse to approximate the shape of the silhouette), we can clearly see the frames of left/right foot strike, given by the (respectively) higher/lower local minima of the graph.  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/imgs/ellipseangles.png" width="50%">  

However, the plot of the ellipse center's height is not visually friendly - this is because in a side treadmill video, the arms and legs cross over the body and each other frequently, skewing the average of all the pixels in the image. Other factors such as clothing and hair also create artifacts in the data.  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/imgs/ellipseheight.png" width="50%">  

## Fitting Data
Using the "vertical position and velocity" technique, position-time series are extracted from 28 subjects. Each subject has a "gallery" and a "probe" series (both consisting of 300 data points). To classify each "probe" series, we directly use the "gallery" videos as a model. We then use nearest-neighbor classifcation, measuring euclidian distance with discrete time warping with hyperparameter w=10.  
Out of the 28 subjects, 15 are correctly classified as the "most likely match".  
If the "correct guess" range is extended to the 5 most likely subjects, our accuracy increases to 93% (26/28).  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/imgs/z_nn_classification.png" width="40%">  

## Average silhouette
To avoid classifying time series data, we can split each subject's data into stride cycles (given by any of the previous methods), then average all of the frames in a cycle to create an image classification problem.    
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/imgs/avgsilhouette.png" width="20%">  