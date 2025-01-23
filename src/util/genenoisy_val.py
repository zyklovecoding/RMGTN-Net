import os
import soundfile as sf
import train_dataset
from tqdm import tqdm
# Initialize the dataset
dataset = train_dataset.TrainDataset(clean_dataset_dir= "/home/doyourthing_zyk/data/vctkdataset/test_clean/",
                                    noise_dataset_dir= "/home/doyourthing_zyk/data/demand_16k/",
                                    snr_range= (0, 12),nchannels=6,ref_channel=5,
                                     selected_channels=[0,1,2,3,4,5])

# Specify the directory where the RIR files are stored
#rir_folder = "/home/GongpingHuang/zyk/McNet-main/rirs/"

# Specify the output directory
output_dir_noisy = "/home/doyourthing_zyk/data/finaldataset/test/noisy/"
output_dir_clean = "/home/doyourthing_zyk/data/finaldataset/test/clean/"
# Ensure the output directory exists
os.makedirs(output_dir_noisy, exist_ok=True)

# Iterate over the dataset
for i in tqdm(range(len(dataset))):
    # Get the clean and noisy signals
    noisy_y, clean_y, clean_file = dataset[i]

    # Get the file name without the extension
    file_name = os.path.splitext(os.path.basename(clean_file))[0]

    # Save the noisy signal to the output directory
    sf.write(os.path.join(output_dir_noisy, f"{file_name}.wav"), noisy_y.T, dataset.sr)
    sf.write(os.path.join(output_dir_clean, f"{file_name}.wav"), clean_y, dataset.sr)