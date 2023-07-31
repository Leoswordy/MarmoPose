import os
# Ignore version incompatibility, not safe, need to be addressed
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["NUMBA_NUM_THREADS"] = "4"
import numexpr as ne
ne.set_num_threads(ne.detect_number_of_cores())


if __name__ == '__main__':
    from marmopose.utils.config import load_config
    config_file = './marmopose/config/config_single_marmoset.toml'
    config = load_config(config_file)

    # from marmopose.calibrate import calibrate
    # from marmopose.utils.plots import plot_calibration
    # calibrate(config, verbose=True)
    # plot_calibration(config, n_show=2)

    # from marmopose.utils.coordinates import set_coordinates
    # set_coordinates(config, obj_name='axes', offset=[0, 0, 0], frame_idx=75)

    # from marmopose.evaluate import evaluate
    # evaluate(config, './labels/models/model_single_v1.1', './labels/original_labels/labels_230703_n669_baseline.slp')

    # from marmopose.predict import predict
    # from marmopose.utils.plots import plot_scores, plot_visibility
    # predict(config, multi_animal = False, verbose = True)
    # plot_scores(config)
    # plot_visibility(config, mode='each')

    # from marmopose.visualize_2d import generate_video_2d
    # generate_video_2d(config, filtered = False, verbose = True)

    # from marmopose.triangulate import triangulate
    # from marmopose.utils.plots import plot_triangulation_errors
    # triangulate(config, filtered = False, verbose = True)
    # plot_triangulation_errors(config)

    # from marmopose.visualize_3d import generate_video_3d
    # generate_video_3d(config)


    from marmopose.realtime.main import realtime_inference
    camera_paths = [
        'rtsp://admin:abc12345@192.168.1.239:554//Streaming/Channels/101', 
        'rtsp://admin:abc12345@192.168.1.247:554//Streaming/Channels/101', 
        'rtsp://admin:abc12345@192.168.1.244:554//Streaming/Channels/101', 
        'rtsp://admin:abc12345@192.168.1.246:554//Streaming/Channels/101'
    ]

    realtime_inference(config, camera_paths=camera_paths, compute_3d=True, 
                       display_3d=True, display_2d=True, display_scale=0.5,
                       crop_size=600, max_queue_size=1500, verbose=True, local_mode=True)
    
    ## Test mac change in main branch



