# human-gait
  
A research project on extracting features of human gait from side-view treadmill videos.

## Vertical position and velocity
To determine vertical position of the subject, we calculate the geometric center of the head. This helps us avoid problems with pixel overlap, since other features will rarely block the head, while the legs and arms frequently cross over each other.  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/10kmposition.png" width="60%">  
Once we have a graph of vertical position, we can take the 1st derivative to measure the subject's vertical velocity. The minimums in the velocity graph signify the frames where the subject's foot strikes the ground. We can then record all the "foot-strike" frames to determine stride cycles.  
<img src="https://raw.githubusercontent.com/imkevinkuo/human-gait/master/10kmvelocity.png" width="60%">  

## Ellipse of Approximation
Method documented in <a href="https://dl.acm.org/citation.cfm?id=2676612">Extracting silhouette-based characteristics for human gait analysis using one camera</a>.  