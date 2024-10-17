from moviepy.editor import ImageSequenceClip
import os

# Path to the directory containing images
image_folder = 'xp_RL'
fps = 5  # frames per second

# Get list of images
images = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]

# Create the video
clip = ImageSequenceClip(images, fps=fps)
clip.write_videofile("output_video.mp4")

