from rename_frames import RenameFrames
import os

curr_path = os.getcwd()

input_folder = os.path.join(curr_path, 'input_test')
output_folder = os.path.join(curr_path, 'output_test')

# input_folder = "C:\Users\Deepak Talwar\Dropbox (Personal)\SJSU\Semesters\Fall2019\CalHacks\badminton-pose-analysis\data_collection\preprocessing\input_test"
# output_folder = "C:\Users\Deepak Talwar\Dropbox (Personal)\SJSU\Semesters\Fall2019\CalHacks\badminton-pose-analysis\data_collection\preprocessing\input_test"

rf = RenameFrames(input_folder, output_folder, "testTournament", "testPlayer", "RAW")

rf.rename_frames()