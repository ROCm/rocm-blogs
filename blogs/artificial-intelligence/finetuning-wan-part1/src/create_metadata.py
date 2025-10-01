import pandas as pd
import os

def create_metadata_csv(videos_file_path, captions_file_path, output_csv_path):
    """
    Create a metadata.csv file combining video filenames with their corresponding captions.
    
    Args:
        videos_file_path: Path to the videos.txt file containing video filenames
        captions_file_path: Path to the prompt.txt file containing captions
        output_csv_path: Path where the metadata.csv file will be saved
    """
    
    # Read video filenames and remove "videos/" prefix
    with open(videos_file_path, 'r', encoding='utf-8') as f:
        video_filenames = [line.strip() for line in f if line.strip()]
    
    # Read captions
    with open(captions_file_path, 'r', encoding='utf-8') as f:
        captions = [line.strip() for line in f if line.strip()]
    
    # Verify that we have the same number of videos and captions
    if len(video_filenames) != len(captions):
        print(f"Warning: Number of video files ({len(video_filenames)}) doesn't match number of captions ({len(captions)})")
        min_length = min(len(video_filenames), len(captions))
        video_filenames = video_filenames[:min_length]
        captions = captions[:min_length]
        print(f'Using first {min_length} entries for both')
    
    # Create DataFrame
    metadata_df = pd.DataFrame({
        'video': video_filenames,
        'prompt': captions
    })
    
    # Save to CSV
    metadata_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f'Created metadata.csv with {len(metadata_df)} entries')
    print(f'Saved to: {output_csv_path}')
    
    # Display first few entries
    print('\nFirst 5 entries:')
    print(metadata_df.head())
    
    return metadata_df

# Define file paths
dataset_dir = './Disney-VideoGeneration-Dataset'
videos_file = os.path.join(dataset_dir, 'videos.txt')
captions_file = os.path.join(dataset_dir, 'prompt.txt')
output_csv = os.path.join(dataset_dir, 'metadata.csv')

# Create the metadata.csv file
metadata_df = create_metadata_csv(videos_file, captions_file, output_csv)