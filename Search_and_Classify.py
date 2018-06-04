# Search the image and find the vehicles in the image
import pickle
import cv2
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip, ImageSequenceClip
from extract_features import *
from heat_map import *
import glob
from multiprocessing import Pool

# load a pe-trained svc model from a serialized (pickle) file
model_name = 'model_linear.p'
model = pickle.load(open(model_name, "rb"))

# get attributes of our svc object
clf = model["svc"]
color_space = model["color_space"]
X_scaler = model["X_scaler"]
orient = model["orient"]
pix_per_cell = model["pix_per_cell"]
cell_per_block = model["cell_per_block"]
spatial_size = model["spatial_size"]
hist_bins = model["hist_bins"]
ystarts = [400, 416, 384, 400, 416, 432, 384, 400, 416, 432, 400, 432]
ystops = [464, 480, 480, 496, 512, 528, 512, 528, 544, 560, 560, 592]
scales = [1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5]


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystarts=ystarts, ystops=ystops, color_space=color_space, scales=scales,
              orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size,
              hist_bins=hist_bins):
    box_list = []
    draw_img = np.copy(img)

    for scale, ystart, ystop in zip(scales, ystarts, ystops):
        img_tosearch = img[ystart:ystop, :, :]

        ctrans_tosearch = np.copy(img_tosearch)
        ctrans_tosearch = covert_color(ctrans_tosearch, color_space)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
                #
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    # cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                    #               (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0,0,255), 3)
                    box_list.append([[xbox_left, ytop_draw + ystart],[xbox_left + win_draw, ytop_draw + win_draw + ystart]])

    return box_list, draw_img

# Combine bounding boxes
def process_img(img):

    # draw_img = np.copy(img)
    box_list, draw_img = find_cars(img)

    heat = np.zeros_like(draw_img[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_img, labels)

    return draw_img

# Process the images from the video
def gen_imgs(file):
    print(file)
    file_name = file.split('\\')[-1]
    img = cv2.imread(file)
    out_img = process_img(img)
    cv2.imwrite('.\\final_imgs\\'+file_name, out_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    # Clip video to image sequence
    clip1 = VideoFileClip("project_video.mp4")
    clip1.write_images_sequence('.\\output_imgs\\frame%04d.png', fps=25, verbose=True, withmask=True, progress_bar=True)
    files = glob.glob('.\\output_imgs\\*.png')
    # files = glob.glob('.\\test_images\\*.jpg'
    #
    # for file in tqdm(files):
    #     print(file)
    #     file_name = file.split('\\')[-1]
    #     img = cv2.imread(file)
    #     out_img = process_img(img)
    #     cv2.imwrite('.\\final_img3\\' + file_name, out_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    # Process the images using multi-processor
    pool = Pool()
    pool.map(gen_imgs, files)

    # Output video from image sequence
    white_output = 'project_video_output.mp4'
    files = glob.glob('.\\final_imgs\\*.png')
    clip2 = ImageSequenceClip(files, fps=25)
    clip2.write_videofile(white_output, audio=False)

    # img = cv2.imread('./test_images/test6.jpg')
    # out_img = process_img(img)
    # plt.imshow(out_img)
    # plt.show()
    #
    # test_out_file = 'test_video_out.mp4'
    # clip_test = VideoFileClip('test_video.mp4')
    # clip_test_out = clip_test.fl_image(process_img)
    # clip_test_out.write_videofile(test_out_file, audio=False)