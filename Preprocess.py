# Extract features from training data and save in pickle file
import glob
from extract_features import *
import pickle


# Read in cars and notcars images
cars_files = glob.glob('.\\vehicles\\**\\*.png')
print(len(cars_files))
notcars_files = glob.glob('.\\non_vehicles\\**\\*.png')
print(len(notcars_files))

color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True# Spatial features on or off
hist_feat = True# Histogram features on or off
hog_feat = True # HOG features on or off

car_features = extract_features(cars_files, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars_files, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,hog_channel=hog_channel,
                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Save feature data
data_name = 'data.p'
pickle.dump({'X_train': X, 'y_train': y}, open(data_name, 'wb'))
