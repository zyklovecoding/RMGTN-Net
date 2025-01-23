from pydub import AudioSegment
import os
from tqdm import tqdm

def segment_wav_files(source_folder, target_folder, segment_duration_ms=2000, count=0):
    # Ensure the target directory exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Iterate through all files in the source folder
    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith(".wav"):
            full_path = os.path.join(source_folder, filename)
            audio = AudioSegment.from_wav(full_path)

            # Calculate the number of segments
            length_ms = len(audio)
            num_segments = length_ms // segment_duration_ms
            if num_segments==0:
                num_segments = 1
            for i in range(num_segments):
                start_ms = i * segment_duration_ms
                end_ms = start_ms + segment_duration_ms

                # Extract segment
                segment = audio[start_ms:end_ms]

                # Calculate the padding needed if any
                padding_needed = segment_duration_ms - len(segment)

                if padding_needed > 0:
                    if padding_needed <= 1500:  # Only pad if less than or equal to 1.5 seconds
                        # Repetitive padding: repeat the segment to fill the gap
                        repeat_times = (padding_needed // len(segment)) + 1
                        padded_segment = (segment * repeat_times)[:padding_needed]  # Repeat and trim to exact padding needed
                        segment += padded_segment
                    else:
                        continue  # Skip this segment if more than 1.5 seconds of padding is needed
                count += 1
                # Save the segment to the target folder
                segment_filename = f"{os.path.splitext(filename)[0]}_{i + 1}.wav"
                segment_path = os.path.join(target_folder, segment_filename)
                segment.export(segment_path, format="wav")

    print("All segments have been processed and saved.")
    return count

# Example usage
source_folder = "/home/doyourthing_zyk/data/vctkdataset/resample_val/"
target_folder = "/home/doyourthing_zyk/data/vctkdataset/test_clean/"
count = segment_wav_files(source_folder, target_folder)
print(count)