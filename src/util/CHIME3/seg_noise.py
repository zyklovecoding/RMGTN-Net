import os
import soundfile as sf
import numpy as np

# Define input and output directories
input_directory = "/home/GongpingHuang/zyk/CHiME-3/CHiME3/data/audio/16kHz/bkrds_combined_unseg/"  # Replace with your input folder path
output_directory = "/home/GongpingHuang/zyk/CHiME-3/CHiME3/data/audio/16kHz/bkrds_combined/"  # Replace with your output folder path

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def slice_wav_files():
    # Iterate over all WAV files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_directory, filename)
            # Read the WAV file
            data, samplerate = sf.read(file_path)
            # data is a numpy array of shape (num_samples, num_channels)
            total_samples = data.shape[0]
            num_channels = data.shape[1] if data.ndim > 1 else 1
            # Compute the length of each segment
            segment_length = total_samples // 100
            if segment_length == 0:
                print(f"File {filename} is too short to be sliced into 50 segments.")
                continue
            # Slice into segments
            for i in range(100):
                start_sample = i * segment_length
                end_sample = start_sample + segment_length if i < 99 else total_samples
                segment_data = data[start_sample:end_sample]
                # Generate output filename
                segment_filename = f"{os.path.splitext(filename)[0]}_segment_{i+1}.wav"
                segment_file_path = os.path.join(output_directory, segment_filename)
                # Write the segment to a new WAV file
                sf.write(segment_file_path, segment_data, samplerate)
            print(f"Sliced {filename} into 100 segments.")

if __name__ == "__main__":
    slice_wav_files()
