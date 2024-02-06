from moviepy.editor import VideoFileClip

def compress_mp4(input_path, output_path, target_size_mb=None, bitrate=None):
    clip = VideoFileClip(input_path)
    if target_size_mb:
        target_bitrate = (target_size_mb * 8 * 1024 * 1024) / clip.duration
        bitrate = f"{int(target_bitrate)}k"
    clip.write_videofile(output_path, bitrate=bitrate)

# Example usage
input_path = '/Users/yash/Documents/spad/img/vids/teaser.mp4'
output_path = 'compressed_video.mp4'
compress_mp4(input_path, output_path, target_size_mb=0.01)  # Compress to target size of 10MB
