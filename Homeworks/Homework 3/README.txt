My solution to the image mosaic problem follows the guide set in the homework PDF going through the processes of: keypoint extraction, keypoint 
matching, fundamental matrix computation, homography determination, and ultimately, image mosaic creation. The process begins key point extraction
 using the Scale-Invariant Feature Transform (SIFT) algorithm, a crucial step for identifying unique features that are invariant to 
 image scale and rotation.

Keypoint matching between images is performed using the FLANN-based matcher, followed by a ratio test to filter out the bad matches based on a 
threshold value of 0.75, This successfully reduces false positives and is able to catch most image mismatches. 
Specifically, a scene is considered the same if there are more than 50 good matches and the match fractions for both images exceed 5%, or if 
there are over 100 good matches with a lower threshold of 2% match fraction. This dual threshold approach allows flexibility in assessing the 
similarity between images, accommodating variations in image quality and content.

Further analysis involves computing the fundamental matrix and using the RANSAC algorithm to identify inliers among the matched keypoints. 
The criteria for a successful match are more stringent now, requiring more than 25 inliers and a 60% percent match of inliers relative to the total 
number, or 75 inliers and a 30 percent match.

The creation of image mosaics is contingent on the successful computation of a homography matrix, with specific thresholds set for the number 
of inliers and their percentage, from the fundamental matrix step, to proceed with mosaic generation. These thresholds ensure that only 
matches with a very high degree of confidence are used, minimizing distortion in the final mosaic.

The mosaic generation itself was created by first making a canvas large enough to accommodate the two images, then applying the homography to the
first image. This transforms it to the coordinates of the second image. The final step is to apply the first image to the second with an alpha blend.
The alpha value I decided on, was .5 as equal balance between the images seems to be the best for determining the quality of the overlay.

The benefits of my approach mainly relate to the fact that multiple threshhold options allow for images matches to be made with either a high number 
of matches or a high percentage of matches without necessarily needing both. Of course a drawback of this is the possibility that a mosaic looks 
bad and "should" have been scrapped. An example on the edge of this is the two colloseum photos. The right side is aligned well, but the left 
looks very messy. 

|Image Pair        | Original Matches | Num Inliers | Percent Inliers | Attempt Mosaic |
|------------------|------------------|-------------|-----------------|----------------|
|VCC1 & VCC3       | 78               | -1          | -1              | -1             |
|VCC1 & VCC2       | 894              | 499         | 55              | 1              |
|VCC1 & Office1    | 42               | -1          | -1              | -1             |
|VCC1 & Office2    | 42               | -1          | -1              | -1             |
|VCC1 & Office3    | 40               | -1          | -1              | -1             |
|VCC1 & Drinks2    | 35               | -1          | -1              | -1             |
|VCC1 & Drinks3    | 44               | -1          | -1              | -1             |
|VCC1 & Drinks1    | 56               | -1          | -1              | -1             |
|VCC1 & Park2      | 20               | -1          | -1              | -1             |
|VCC1 & Tree4      | 42               | -1          | -1              | -1             |
|VCC1 & Park1      | 16               | -1          | -1              | -1             |
|VCC1 & Tree2      | 34               | -1          | -1              | -1             |
|VCC1 & Tree3      | 35               | -1          | -1              | -1             |
|VCC1 & Tree1      | 24               | -1          | -1              | -1             |
|VCC3 & VCC2       | 100              | -1          | -1              | -1             |
|VCC3 & Office1    | 18               | -1          | -1              | -1             |
|VCC3 & Office2    | 15               | -1          | -1              | -1             |
|VCC3 & Office3    | 10               | -1          | -1              | -1             |
|VCC3 & Drinks2    | 12               | -1          | -1              | -1             |
|VCC3 & Drinks3    | 18               | -1          | -1              | -1             |
|VCC3 & Drinks1    | 13               | -1          | -1              | -1             |
|VCC3 & Park2      | 8                | -1          | -1              | -1             |
|VCC3 & Tree4      | 11               | -1          | -1              | -1             |
|VCC3 & Park1      | 8                | -1          | -1              | -1             |
|VCC3 & Tree2      | 9                | -1          | -1              | -1             |
|VCC3 & Tree3      | 15               | -1          | -1              | -1             |
|VCC3 & Tree1      | 16               | -1          | -1              | -1             |
|VCC2 & Office1    | 81               | -1          | -1              | -1             |
|VCC2 & Office2    | 70               | -1          | -1              | -1             |
|VCC2 & Office3    | 55               | -1          | -1              | -1             |
|VCC2 & Drinks2    | 48               | -1          | -1              | -1             |
|VCC2 & Drinks3    | 58               | -1          | -1              | -1             |
|VCC2 & Drinks1    | 62               | -1          | -1              | -1             |
|VCC2 & Park2      | 35               | -1          | -1              | -1             |
|VCC2 & Tree4      | 46               | -1          | -1              | -1             |
|VCC2 & Park1      | 29               | -1          | -1              | -1             |
|VCC2 & Tree2      | 46               | -1          | -1              | -1             |
|VCC2 & Tree3      | 31               | -1          | -1              | -1             |
|VCC2 & Tree1      | 33               | -1          | -1              | -1             |
|Office1 & Office2 | 229              | 162         | 70              | 1              |
|Office1 & Office3 | 128              | 54          | 42              | 1              |
|Office1 & Drinks2 | 15               | -1          | -1              | -1             |
|Office1 & Drinks3 | 20               | -1          | -1              | -1             |
|Office1 & Drinks1 | 15               | -1          | -1              | -1             |
|Office1 & Park2   | 11               | -1          | -1              | -1             |
|Office1 & Tree4   | 50               | -1          | -1              | -1             |
|Office1 & Park1   | 11               | -1          | -1              | -1             |
|Office1 & Tree2   | 38               | -1          | -1              | -1             |
|Office1 & Tree3   | 47               | -1          | -1              | -1             |
|Office1 & Tree1   | 30               | -1          | -1              | -1             |
|Office2 & Office3 | 97               | 45          | 46              | 1              |
|Office2 & Drinks2 | 11               | -1          | -1              | -1             |
|Office2 & Drinks3 | 13               | -1          | -1              | -1             |
|Office2 & Drinks1 | 15               | -1          | -1              | -1             |
|Office2 & Park2   | 13               | -1          | -1              | -1             |
|Office2 & Tree4   | 39               | -1          | -1              | -1             |
|Office2 & Park1   | 10               | -1          | -1              | -1             |
|Office2 & Tree2   | 26               | -1          | -1              | -1             |
|Office2 & Tree3   | 46               | -1          | -1              | -1             |
|Office2 & Tree1   | 36               | -1          | -1              | -1             |
|Office3 & Drinks2 | 19               | -1          | -1              | -1             |
|Office3 & Drinks3 | 19               | -1          | -1              | -1             |
|Office3 & Drinks1 | 22               | -1          | -1              | -1             |
|Office3 & Park2   | 7                | -1          | -1              | -1             |
|Office3 & Tree4   | 29               | -1          | -1              | -1             |
|Office3 & Park1   | 11               | -1          | -1              | -1             |
|Office3 & Tree2   | 11               | -1          | -1              | -1             |
|Office3 & Tree3   | 26               | -1          | -1              | -1             |
|Office3 & Tree1   | 22               | -1          | -1              | -1             |
|Drinks2 & Drinks3 | 280              | 73          | 26              | 1              |
|Drinks2 & Drinks1 | 327              | 51          | 15              | 1              |
|Drinks2 & Park2   | 30               | -1          | -1              | -1             |
|Drinks2 & Tree4   | 48               | -1          | -1              | -1             |
|Drinks2 & Park1   | 73               | -1          | -1              | -1             |
|Drinks2 & Tree2   | 62               | -1          | -1              | -1             |
|Drinks2 & Tree3   | 75               | -1          | -1              | -1             |
|Drinks2 & Tree1   | 36               | -1          | -1              | -1             |
|Drinks3 & Drinks1 | 100              | -1          | -1              | -1             |
|Drinks3 & Park2   | 24               | -1          | -1              | -1             |
|Drinks3 & Tree4   | 50               | -1          | -1              | -1             |
|Drinks3 & Park1   | 75               | -1          | -1              | -1             |
|Drinks3 & Tree2   | 60               | -1          | -1              | -1             |
|Drinks3 & Tree3   | 74               | -1          | -1              | -1             |
|Drinks3 & Tree1   | 53               | -1          | -1              | -1             |
|Drinks1 & Park2   | 38               | -1          | -1              | -1             |
|Drinks1 & Tree4   | 46               | -1          | -1              | -1             |
|Drinks1 & Park1   | 51               | -1          | -1              | -1             |
|Drinks1 & Tree2   | 42               | -1          | -1              | -1             |
|Drinks1 & Tree3   | 57               | -1          | -1              | -1             |
|Drinks1 & Tree1   | 43               | -1          | -1              | -1             |
|Park2 & Tree4     | 3                | -1          | -1              | -1             |
|Park2 & Park1     | 151              | 148         | 98              | 1              |
|Park2 & Tree2     | 5                | -1          | -1              | -1             |
|Park2 & Tree3     | 5                | -1          | -1              | -1             |
|Park2 & Tree1     | 1                | -1          | -1              | -1             |
|Tree4 & Park1     | 33               | -1          | -1              | -1             |
|Tree4 & Tree2     | 207              | 87          | 42              | 1              |
|Tree4 & Tree3     | 616              | 455         | 73              | 1              |
|Tree4 & Tree1     | 99               | -1          | -1              | -1             |
|Park1 & Tree2     | 5                | -1          | -1              | -1             |
|Park1 & Tree3     | 5                | -1          | -1              | -1             |
|Park1 & Tree1     | 6                | -1          | -1              | -1             |
|Tree2 & Tree3     | 1121             | 876         | 78              | 1              |
|Tree2 & Tree1     | 1427             | 1236        | 86              | 1              |
|Tree3 & Tree1     | 425              | 200         | 47              | 1              |
----------------------------------------------------------------------------------------