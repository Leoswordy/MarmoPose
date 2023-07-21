import numpy as np
from multiprocessing import Queue, Event
from typing import Dict

from marmopose.realtime.processor import DisplayProcess


class EventControl:
    """
    Controls events based on data from queues.

    Attributes:
        bodyparts_dict (dict): Body parts dictionary.
        points_3d_queue (Queue): Queue for 3D points.
        points_2d_queue (Queue): Queue for 2D points.
        images_2d_queue (Queue): Queue for 2D images.
        stop_event (Event): Event to stop the processing.

        display (bool): Display flag.
        verbose (bool): Verbose flag.
        display_scale (float): Scale for display.
        display_queue (Queue): Queue for display data.
        display_process (DisplayProcess): Display process.
    """
    def __init__(self, config: Dict, points_3d_queue: Queue, points_2d_queue: Queue, images_2d_queue: Queue, 
                 stop_event: Event, display_3d: bool, display_2d: bool, display_scale: float, verbose: bool):
        self.bodyparts_dict = {bp: idx for idx, bp in enumerate(config['bodyparts'])}
        self.points_3d_queue = points_3d_queue
        self.points_2d_queue = points_2d_queue
        self.images_2d_queue = images_2d_queue
        self.stop_event = stop_event

        self.display = display_3d or display_2d
        self.verbose = verbose

        if self.display:
            self.display_scale = display_scale
            self.display_queue = Queue(maxsize=1500)
            self.display_process = DisplayProcess(config, self.display_queue, stop_event, 
                                                  display_3d, display_2d, display_scale, verbose)
            self.display_process.start()
    
    def run(self):
        """Run the event control process."""
        while not self.stop_event.is_set():
            if not self.points_2d_queue.empty() and not self.images_2d_queue.empty() and not self.points_3d_queue.empty():
                # all_points_2d shape of (n_tracks, n_cams x 1, n_bodyparts, 2)
                # all_scores_2d shape of (n_tracks, n_cams x 1, n_bodyparts)
                # all_points_3d shape of (n_tracks, n_frames=1, n_bodyparts, 3)
                # all_scores_3d shape of (n_tracks, n_frames=1, n_bodyparts)
                all_points_2d, all_scores_2d = self.points_2d_queue.get()
                images_2d = self.images_2d_queue.get()
                all_points_3d, all_scores_3d = self.points_3d_queue.get()

                self.control(all_points_2d, all_scores_2d, all_points_3d, all_scores_3d)

                if self.display:
                    self.display_queue.put((all_points_3d, images_2d))

        print('Event Control Finished!')

    def control(self, all_points_2d: np.ndarray, all_scores_2d: np.ndarray, all_points_3d: np.ndarray, all_scores_3d: np.ndarray):
        """
        This function should be overrided to implement your control purpose.
        
        Args:
            all_points_2d: All 2D points. Shape of (n_tracks, n_cams x 1, n_bodyparts, 2)
            all_scores_2d: All 2D scores. Shape of (n_tracks, n_cams x 1, n_bodyparts)
            all_points_3d: All 3D points. Shape of (n_tracks, n_frames=1, n_bodyparts, 3)
            all_scores_3d: All 3D scores. Shape of (n_tracks, n_frames=1, n_bodyparts)
        """
        # Template code for dealing with 3D coordinates
        if all_points_3d is not None:
            points_3d_track1 = all_points_3d[0, 0] #(n_bodyparts, 3)

            # Get any bodypart of interest
            target_bodypart = 'head'
            target_3d_position = points_3d_track1[self.bodyparts_dict[target_bodypart]] #(x, y, z)

            # Or get the average of all the bodyparts
            avg_3d_position = np.nanmean(points_3d_track1, axis=0)

            if avg_3d_position[1] > 400:
                print('On the right side!!!')
            if avg_3d_position[2] > 300:
                print('Jumping!!!')
        
        # Deal with 2D coordinates
        points_2d_track1_cam1 = all_points_2d[0, 0] # (n_bodyparts, 2)
        # Any other operations ...
