import time
import numpy as np
from multiprocessing import Queue, Event
from typing import Dict, Any

from marmopose.realtime.data_processor import DisplayProcess


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
    def __init__(self, 
                 config: Dict[str, Any], 
                 points_3d_queue: Queue, 
                 points_2d_queue: Queue, 
                 images_2d_queue: Queue, 
                 stop_event: Event, 
                 display_3d: bool = True, 
                 display_2d: bool = True, 
                 display_scale: float = 0.5, 
                 verbose: bool = True):
        self.points_3d_queue = points_3d_queue
        self.points_2d_queue = points_2d_queue
        self.images_2d_queue = images_2d_queue
        self.stop_event = stop_event

        self.bodyparts_dict = {bp: idx for idx, bp in enumerate(config['animal']['bodyparts'])}

        self.display = display_3d or display_2d
        self.verbose = verbose

        if self.display:
            self.display_scale = display_scale
            self.display_queue = Queue(maxsize=1500)
            self.display_process = DisplayProcess(config=config, 
                                                  display_queue=self.display_queue, 
                                                  stop_event=stop_event,
                                                  display_3d=display_3d,
                                                  display_2d=display_2d, 
                                                  display_scale=display_scale, 
                                                  verbose=verbose)
            self.display_process.start()
    
    def run(self):
        while not self.stop_event.is_set():
            if not self.points_2d_queue.empty() and not self.images_2d_queue.empty() and not self.points_3d_queue.empty():
                # all_points_with_score_2d shape of (n_cams, n_tracks, n_bodyparts, (x,y,score)))
                # all_points_3d shape of (n_tracks, n_frames=1, n_bodyparts, (x,y,z)))
                all_points_3d = self.points_3d_queue.get()
                all_points_with_score_2d = self.points_2d_queue.get()
                images_2d = self.images_2d_queue.get()

                self.control(all_points_with_score_2d, all_points_3d)

                if self.display:
                    self.display_queue.put((all_points_3d, images_2d))
            time.sleep(0.001) # Avoid busy waiting

        print('Event Control Finished!')

    def control(self, all_points_with_score_2d: np.ndarray, all_points_3d: np.ndarray):
        """Controls events based on the lateset 2d and 3d points.
        This function should be overrided to implement your control purpose.
        
        Args:
            all_points_with_score_2d: All 2D points. Shape of (n_cams, n_tracks, n_bodyparts, (x,y,score)))
            all_points_3d: All 3D points. Shape of (n_tracks, n_frames=1, n_bodyparts, (x,y,z)))
        """
        # Template code for dealing with 3D coordinates
        if all_points_3d is not None:
            points_3d_track1 = all_points_3d[0, 0] #(n_bodyparts, 3)

            # Get any bodypart of interest
            target_bodypart = 'head'
            target_3d_position = points_3d_track1[self.bodyparts_dict[target_bodypart]] #(x, y, z)

            # Or get the average of all the bodyparts
            avg_3d_position = np.nanmean(points_3d_track1, axis=0)

            # if avg_3d_position[1] > 400:
            #     print('On the right side!!!')
            # if avg_3d_position[2] > 300:
            #     print('Jumping!!!')
        
        # Deal with 2D coordinates
        points_2d_track1_cam1 = all_points_with_score_2d[0, 0] # (n_bodyparts, 3)
        # Any other operations ...
