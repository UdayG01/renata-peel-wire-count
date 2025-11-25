import cv2
import os

class VideoFrameExtractor:
    def __init__(self, video_path, output_dir):
        """
        Initialize the VideoFrameExtractor.

        Parameters:
        - video_path (str): Path to the input video file.
        - output_dir (str): Directory to save the extracted frames.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_count = 0

    def load_video(self):
        """
        Load the video using OpenCV.

        Returns:
        - cv2.VideoCapture: Video capture object.
        """
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        return cv2.VideoCapture(self.video_path)

    def extract_frames(self, frame_rate=5):
        """
        Extract frames from the video at the specified frame rate.

        Parameters:
        - frame_rate (int): Number of frames to extract per second.

        Returns:
        - List[str]: List of paths to the saved frames.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the video
        cap = self.load_video()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frame_rate)  # Interval between frames to extract

        frame_paths = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame if it's within the extraction interval
            if self.frame_count % frame_interval == 0:
                frame_filename = f"frame_{self.frame_count}.jpg"
                frame_path = os.path.join(self.output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)

            self.frame_count += 1

        cap.release()
        return frame_paths

    def get_total_frames(self):
        """
        Get the total number of frames extracted.

        Returns:
        - int: Total number of frames extracted.
        """
        return len(os.listdir(self.output_dir))


def main():
    # Input parameters
    video_path = "assets/input videos/Sample input video.mp4"  # Replace with your video path
    output_dir = "assets/output frames"  # Directory to save frames
    frame_rate = 5  # Extract 5 frames per second

    # Create an instance of VideoFrameExtractor
    extractor = VideoFrameExtractor(video_path, output_dir)

    # Extract frames
    extracted_frame_paths = extractor.extract_frames(frame_rate)

    # Print results
    print(f"Total frames extracted: {len(extracted_frame_paths)}")
    print("Extracted frames saved to:", output_dir)


if __name__ == "__main__":
    main()