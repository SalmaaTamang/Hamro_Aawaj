
import os
import cv2
from datetime import datetime

class VideoRecorderPipeline:
    def __init__(self, output_base_folder='frames', frame_width=1920, frame_height=1080, capture_device=0):
        self.output_base_folder = output_base_folder
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.capture_device = capture_device
        self.video = None
        self.output_folder = None
        self.frames = []
        self.recording = False

    def create_output_folder(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(self.output_base_folder, f"session_{timestamp}")
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Created output folder: {self.output_folder}")

    def initialize_video_capture(self):
        try:
            self.create_output_folder()
            self.video = cv2.VideoCapture(self.capture_device)
            if not self.video.isOpened():
                raise IOError(f"Unable to open video capture device {self.capture_device}")
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            print(f"Successfully opened video capture device: {self.capture_device}")
        except Exception as e:
            print(f"Error initializing video capture: {e}")

    def start_recording(self):
        try:
            if self.video is None:
                self.initialize_video_capture()

            i = 1
            wait = 0

            while True:
                ret, img = self.video.read()
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(img, f'Frames Captured: {i-1}', (20, 40), font, 2, (255, 0,0), 2, cv2.LINE_AA)

                cv2.imshow('live video', img)
                key = cv2.waitKey(100)
                wait += 100
                if key == ord('q'):
                    break
                if wait == 3000:
                    filename = os.path.join(self.output_folder, f'Frame_{i}.jpg')
                    cv2.imwrite(filename, img)
                    self.frames.append(img)
                    i += 1
                    wait = 0
        except Exception as e:
            print(f"Error during recording: {e}")


    def stop_recording(self):
        try:
            if self.video is not None:
                self.recording = False
                self.video.release()
                self.video = None
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error stopping recording: {e}")

    def get_captured_frames(self):
        captured_frame = self.frames.copy()
        self.frames.clear()
        return captured_frame
