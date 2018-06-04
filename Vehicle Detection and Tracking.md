# Vehicle Detection and Tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:
- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
- Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

---

### Color Space and Histogram of Oriented Gradients (HOG)
The code for this step is contained in the `Preprocess.py` and `train_model.py`.
#### 1. Examples of the training images

I started by reading in all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![p1]()

#### 2. Exploration of color space
I then expolred different color spaces (RGB, HSV, LUV, HLS) and here is an example of the distribution of color values in a car image.

![p4]()

#### 3. Exploration of HOG
Next, I explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) to extract different HOG features. I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using grey scale and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![p5]()

#### 4. Final choice of color space and HOG parameters
 After tried varoius combinations of parameters, I decided to choose extraction parameters shown as following:
 |Parameters|Value|
 |----------|-----|
 |Using Spatial Features|Yes|
 |Using Histogram Features|Yes|
 |Using HOG Features|Yes|
 |Color Space|HSV|
 |Spatial Binning Dimensions|16×16|
 |Number of histogram bins|16|
 |HOG Channel|ALL|
 |HOG orient|11|
 |HOG pixel per cell|8|
 |HOG cell per block|2|
I use `cv2.imread()` to read images which generates `BGR` channel images. So, I first convert the color channel to `HSV` and then I extract the spatial features and the histogram features using the functions provided in the course. Next I extract HOG features from all of the three channels. Finally, I concatenate these three features together as a feature vector of a training example.

#### 5. Training a classifier
I combined a non-linear SVM and a random forest classifier using the extracted features with spatial features, histogram features and HOG features. 

### Sliding Window Search
The code for this step is contained in the `Search_and_Classify.py`.

#### 1. Implement the sliding window search

I used `find_cars()` function to implement the sliding window search and make a test on the image in sliding window to find windows containing a car. First, if the `scale` is not 1, I resized the image to original image shape devided by scale which means I used fixed size of window to slide on different scales of images. Then I set the window size to 64×64 which is the size of the training images and the cells per step to 2 instead of using overlap rate. Next I extract the HOG features of the entire image and for each sliding step I subsample the HOG fetures from the entire HOG matrix. Finally, I extract the spatial features and the histogram features of the overlapped region and put them together to make a prediction.

I used different combinations of start, stop positons in y axis and scales to achieve satisfied performance. Here is the combinations:
|y_start| y_stop|scale|
|------|------|-----|
|400|464|1.0|
|416|480|1.0|
|384|480|1.5|
|400|496|1.5|
|416|512|1.5|
|432|528|1.5|
|384|512|2.0|
|400|528|2.0|
|416|544|2.0|
|432|560|2.0|
|400|560|2.5|
|432|592|2.5|
Here is an example of the sliding windows:

![p6]()

When scale=1.0, the classifier produces many false positives and when scale is larger than 2.5, the window is too large for locating a car. And it shows we can get most true positives at 1.5 and 2.0 of the scale.

#### 2. Examples of test images
Here are some examples of the result after finding the cars in test images:

![p7]()

### Video Implementation

#### 1. Final video link

Here's a [link]() to my video result.

#### 2. Filter false positive redult and combine bounding boxes

The code for this step is contained in the `heat_map.py` and `process_img()` function in `Search_and_Classify.py`.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.
Here's an example result showing the heatmap from a series of frames of video and the result of `scipy.ndimage.measurements.label()`:

![p8]()

Here the resulting bounding boxes are drawn onto the last frame in the series:

![p9]()

### Discussion

The pipeline might fail because of the oncoming cars and light conditions. When I achieve a high accuracy classifier, the oncoming cars are usually detected. And the tree shadow might also be classified as a car. The detector usually failed when the aimed cars are shown in similar color background (e.g. a black car in black background). I think the possible improvement is using more specific features to detect the cars rather than the combination of colors.