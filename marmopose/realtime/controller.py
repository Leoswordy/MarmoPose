import time
import logging
import numpy as np
from multiprocessing import Queue

from marmopose.config import Config
from marmopose.realtime.data_processor import DisplayProcess
from marmopose.processing.autoencoder import PoseProcessor
from marmopose.utils.helpers import drain_queue
from marmopose.utils.data_io import init_appendable_h5, save_data_online_h5

logger = logging.getLogger(__name__)


class EventControl:
    """
    Controls events based on data from queues.

    Attributes:
        bodyparts_dict (dict): Body parts dictionary.
        data_queue (Queue): Queue for a dict of 2d points, bboxes, 3d points, and images.
        stop_event (Event): Event to stop the processing.

        display (bool): Display flag.
        display_scale (float): Scale for display.
        display_queue (Queue): Queue for display data.
        display_process (DisplayProcess): Display process.
    """
    def __init__(self, 
                 config: Config, 
                 data_queue: Queue,
                 stop_event, 
                 display_3d: bool = True, 
                 display_2d: bool = False, 
                 display_scale: float = 1.0):
        self.config = config
        self.data_queue = data_queue
        self.stop_event = stop_event

        self.bodyparts_dict = {bp: idx for idx, bp in enumerate(config.animal['bodyparts'])}
        self.poseprocessor = PoseProcessor(config.animal['bodyparts'])

        self.display_2d = display_2d
        self.display = display_3d or display_2d

        if self.display:
            self.display_scale = display_scale
            self.display_queue = Queue()
            self.display_process = DisplayProcess(config=config, 
                                                  display_queue=self.display_queue, 
                                                  stop_event=stop_event,
                                                  display_3d=display_3d,
                                                  display_2d=display_2d, 
                                                  display_scale=display_scale)
            self.display_process.start()
        
        init_appendable_h5(config)
    
    def run(self):
        while not self.stop_event.is_set():
            if not self.data_queue.empty():
                data = self.data_queue.get(timeout=5)

                save_data_online_h5(self.config, data['points_2d'], data['bboxes'], data['points_3d'])
                self.control(data['points_2d'], data['bboxes'], data['points_3d'])

                if self.display:
                    self.display_queue.put(data)
            else:
                time.sleep(0.001) # Avoid busy waiting

        logger.info(f'2D and 3D coordinates saved in {self.config.project_path}')
        self.clean()
    
    def clean(self):
        self.display_process.join()

        drain_queue(self.data_queue)
        self.data_queue.close()
        self.data_queue.join_thread()

        drain_queue(self.display_queue)
        self.display_queue.close()
        self.display_queue.join_thread()

        logger.info("Event Control Finished")
    
    def control(self, all_points_with_score_2d: np.ndarray, all_bboxes: np.ndarray, all_points_3d: np.ndarray):
        """Controls events based on the lateset 2d and 3d points.
        This function should be overrided to implement your control purpose.
        
        Args:
            all_points_with_score_2d: All 2D points. Shape of (n_cams, n_tracks, n_bodyparts, (x,y,score)))
            all_bboxes: All bounding boxes. Shape of (n_cams, n_tracks, (x1, y1, x2, y2))
            all_points_3d: All 3D points. Shape of (n_tracks, n_frames=1, n_bodyparts, (x,y,z)))
        """
        # Template code for dealing with 3D coordinates
        # if all_points_3d is not None:
        #     pass
            # ========== Complex detection ==========
            # signal = self.detect_watch_left(all_points_3d[0])
            # if signal:
            #     print('Watch left!!!')

            # ========== Get any bodypart of interest ==========
            # points_3d_track1 = all_points_3d[0, 0] #(n_bodyparts, 3)
            # target_bodypart = 'head'
            # target_3d_position = points_3d_track1[self.bodyparts_dict[target_bodypart]] #(x, y, z)

            # ========== Or get the average of all the bodyparts ==========
            # avg_3d_position = np.nanmean(points_3d_track1, axis=0)

            # if avg_3d_position[1] > 400:
            #     print('On the right side!!!')
            # if avg_3d_position[2] > 300:
            #     print('Jumping!!!')
        
        # ========== Deal with 2D coordinates ==========
        points_2d_track1_cam1 = all_points_with_score_2d[0, 0] # (n_bodyparts, 3)
        # Any other operations ...
    
    def detect_watch_left(self, points_3d):
        normalized_pose = self.poseprocessor.normalize(points_3d)[0] # (n_bodyparts, 3)

        head = normalized_pose[self.bodyparts_dict['head']]
        mid = (normalized_pose[self.bodyparts_dict['leftear']] + normalized_pose[self.bodyparts_dict['rightear']])/2
        direction = head - mid


        N = np.array([0, 1, 0]) 
        V_normalized = direction / np.linalg.norm(direction)
        N_normalized = N / np.linalg.norm(N)

        cos_theta = np.dot(V_normalized, N_normalized)
        theta = np.arccos(cos_theta)

        angle_with_plane = np.pi/2 - theta
        angle_with_plane_degrees = np.degrees(angle_with_plane)

        if angle_with_plane_degrees > 45:
            return True
        return False

