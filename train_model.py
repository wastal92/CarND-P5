# Train classifier and save
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins

# Read in training data
data_name = 'data1.p'
data = pickle.load(open(data_name, 'rb'))
X = data['X_train']
y = data['y_train']

# Split the train data
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('train data length:', X_train.shape[0])
print('Feature vector length:', X_train.shape[1])
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
test_score = round(svc.score(X_test, y_test), 4)
print('Test Accuracy of SVC = ', test_score)
# Check the prediction time for a single sample
t=time.time()

# Save the model
model_name = 'model_linear.p'
pickle.dump({
    'svc': svc,
    'color_space': color_space,
    'orient': orient,
    'pix_per_cell': pix_per_cell,
    'cell_per_block': cell_per_block,
    'spatial_size' : spatial_size,
    'hist_bins' : hist_bins,
    'X_scaler' : X_scaler,
    'hog_channel': hog_channel
}, open(model_name, 'wb'))
print('Model saved in model.p')