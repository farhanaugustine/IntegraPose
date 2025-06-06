import cv2
import threading
import os
import time

class WebcamManager:
    """
    Manages capturing a raw video feed from a webcam in a separate thread.
    This prevents the main GUI from freezing during capture.
    """
    def __init__(self, camera_index, save_dir="raw_captures"):
        """
        Initializes the WebcamManager.

        Args:
            camera_index (int): The index of the camera to use (e.g., 0).
            save_dir (str): The directory where the raw capture will be saved.
        """
        self.camera_index = camera_index
        self.save_dir = save_dir
        
        # I'll create the save directory if it doesn't exist.
        os.makedirs(self.save_dir, exist_ok=True)
        
        # These will be initialized when capture starts.
        self.capture = None
        self.video_writer = None
        self.capture_thread = None
        
        # This flag controls the capture loop in the thread.
        self.is_running = False
        
        # This event signals that the setup in the thread is complete.
        self._setup_complete = threading.Event()
        self._setup_success = False

    def start_capture(self):
        """
        Starts the webcam capture in a new thread.
        Returns True if setup was successful, False otherwise.
        """
        if self.is_running:
            print("Capture is already running.")
            return True

        self.is_running = True
        self._setup_complete.clear() # Reset event for this run
        
        # I am creating a daemon thread so it automatically exits if the main app closes unexpectedly.
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # This ensures the camera is open and the writer is ready before the method returns.
        setup_finished_in_time = self._setup_complete.wait(timeout=5.0) # 5-second timeout for camera to open
        
        if not setup_finished_in_time:
            print("Error: Webcam setup timed out.")
            self.is_running = False # Stop the loop if it's stuck
            return False
            
        return self._setup_success

    def _capture_loop(self):
        """The main loop that runs in the thread to capture and write frames."""
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                print(f"Error: Could not open camera with index {self.camera_index}.")
                self._setup_success = False
                self._setup_complete.set()
                return

            # Get video properties from the camera
            frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            
            # If FPS is 0, default to 30 to avoid errors.
            if fps == 0:
                print("Warning: Camera FPS is 0, defaulting to 30.")
                fps = 30

            # Define filename and video writer object
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(self.save_dir, f"raw_capture_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4 files
            self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

            if not self.video_writer.isOpened():
                print(f"Error: Could not open video writer for path {save_path}.")
                self._setup_success = False
                self.capture.release()
                self._setup_complete.set()
                return

            print(f"Raw capture started. Saving to: {save_path}")
            self._setup_success = True
            self._setup_complete.set() # Signal that setup is done

            # The actual capture loop
            while self.is_running:
                ret, frame = self.capture.read()
                if not ret:
                    # If we can't read a frame, stop the loop.
                    print("Error: Could not read frame from camera. Stopping capture.")
                    break
                self.video_writer.write(frame)

        except Exception as e:
            print(f"An error occurred in the capture thread: {e}")
            self._setup_success = False
        finally:
            # This block ensures that resources are always released.
            if self.capture:
                self.capture.release()
            if self.video_writer:
                self.video_writer.release()
            self._setup_complete.set() # Ensure this is always set
            print("Raw capture resources released.")

    def stop_capture(self):
        """Stops the webcam capture thread and waits for it to finish."""
        if not self.is_running:
            return

        # Setting this to False will cause the _capture_loop to exit.
        self.is_running = False
        
        # Wait for the thread to complete its execution (i.e., finish writing and release resources).
        self.capture_thread.join(timeout=2.0) # Add a timeout
        if self.capture_thread.is_alive():
            print("Warning: Capture thread did not exit gracefully.")
            
        print("Raw capture stopped.")