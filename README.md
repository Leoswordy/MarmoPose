# Maintain
- previous_offset: (n_cams, n_tracks, xy)
- could crop it with (x, y, crop_size, crop_size)
- crops: (n_cams, n_tracks, crop_size, crop_size, n_channels)


# Read image
- images: (n_cams, width, height, channel)

# Preprocess
## Single
- If previous_offset of a cam is none (nan), predict all the whole image(to be add a centorid model?)
## Multi
- If any previous_offset of a cam if none, repredict the roi of the cam, get offsets
- Combine it with other prev_offset, get crops and pass to the conf model

# Predict
- Pred: (n_cams, n_tracks, n_bodyparts, (x, y, score))

# Postprocess points
- Add Pred and previous_offset, get the true coordinates

# Postprocess previous_offset
- If track not found, set it to nan, else update offset (Padding or move???)

# Label image
- Label image with preds

# Triangulate
- swap points_2d (0, 1) to (n_tracks, n_cams, n_bodyparts, 3)
- Add new axis in the for loop: (n_cams, n_frames=1. n_bodyparts, 3)
- triangulate result (n_frames=1, n_bodyparts, 3)
- merge (n_tracks, n_frames=1, n_bodyparts, 3)
