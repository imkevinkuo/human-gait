# human-gait
  
A research project on extracting features of human gait from side-view treadmill videos.  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/00094.png" width="20%">
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/00001red.png" width="20%">

## Vertical position and velocity
To determine vertical position of the subject, we calculate the geometric center of the head. This helps us avoid problems with pixel overlap, since other features will rarely block the head, while the legs and arms frequently cross over each other.  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/10kmposition.png" width="60%">  
Once we have a graph of vertical position, we can take the 1st derivative to measure the subject's vertical velocity. The minimums in the velocity graph signify the frames where the subject's foot strikes the ground. We can then record all the "foot-strike" frames to determine stride cycles.  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/10kmvelocity.png" width="60%">  

## Ellipse of Approximation
Method documented in <a href="https://dl.acm.org/citation.cfm?id=2676612">Extracting silhouette-based characteristics for human gait analysis using one camera</a>.  
If we plot the angle of inclination (obtained by using an ellipse to approximate the shape of the silhouette), we can clearly see the frames of left/right foot strike, given by the (respectively) higher/lower local minima of the graph.  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/ellipseangles.png" width="60%">  