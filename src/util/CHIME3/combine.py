import os
import wave
import numpy as np

# Define base directories
source_directory = "/home/doyourthing_zyk/data"
noisy_target_directory = "/home/doyourthing_zyk/data/chimenoise/"

def combine_wav_files(folder_name, filenames):
    # Create a dictionary to group files by their file ID (excluding the channel part)
    file_groups = {}
    for filename in filenames:
        # Extract file_id and channel information
        parts = filename.split('.')
        if len(parts) < 3:
            continue  # Skip files that don't have the expected format
        file_id = '.'.join(parts[:-2])
        channel = parts[-2]  # The channel number part, e.g., 'CH1'
        if file_id not in file_groups:
            file_groups[file_id] = {}
        file_groups[file_id][channel] = os.path.join(source_directory, folder_name, filename)

    # Process each group of files
    for file_id, channels in file_groups.items():
        channel_data = []
        sample_width = None
        frame_rate = None
        n_frames = None
        missing_channel = False

        # Ensure all channels from CH1 to CH6 are present
        for ch in range(6):
            ch_key = f'CH{ch+1}'
            if ch_key in channels:
                with wave.open(channels[ch_key], 'rb') as wf:
                    if sample_width is None:
                        sample_width = wf.getsampwidth()
                        frame_rate = wf.getframerate()
                        n_frames = wf.getnframes()
                    else:
                        # Ensure all parameters match
                        if (wf.getsampwidth() != sample_width or
                            wf.getframerate() != frame_rate or
                            wf.getnframes() != n_frames):
                            print(f"Audio parameters do not match for {file_id} channel {ch_key}")
                            missing_channel = True
                            break
                    frames = wf.readframes(n_frames)
                    # Convert frames to numpy array
                    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
                    channel_data.append(np.frombuffer(frames, dtype=dtype))
            else:
                print(f"Missing channel {ch_key} for {file_id}, skipping.")
                missing_channel = True
                break

        if not missing_channel:
            # Stack channels and interleave samples
            multi_channel_data = np.vstack(channel_data).T  # Transpose to get samples in rows
            interleaved = multi_channel_data.flatten().tobytes()
            # Define target directory
            target_directory = noisy_target_directory
            if not os.path.exists(target_directory):
                os.makedirs(target_directory, exist_ok=True)
            combined_file_path = os.path.join(target_directory, f"{file_id}_combined.wav")
            # Write multi-channel wav file
            with wave.open(combined_file_path, 'wb') as wf:
                wf.setnchannels(len(channel_data))
                wf.setsampwidth(sample_width)
                wf.setframerate(frame_rate)
                wf.writeframes(interleaved)

def process_folders():
    for folder_name in os.listdir(source_directory):
        if folder_name.startswith("backgrounds"):
            folder_path = os.path.join(source_directory, folder_name)
            # Get all wav files in the folder
            wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
            combine_wav_files(folder_name, wav_files)

if __name__ == "__main__":
    process_folders()
