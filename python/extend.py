import imageio
import numpy as np

video_path = '/Users/yash/Documents/spad/img/vids/teaser.mp4'
output_path = 'extended_teaser.mp4'

reader = imageio.get_reader(video_path)
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer(output_path, fps=fps)

for frame in reader:
    writer.append_data(frame)

last_frame = frame
for _ in range(int(fps) * 2):
    writer.append_data(last_frame)

writer.close()
