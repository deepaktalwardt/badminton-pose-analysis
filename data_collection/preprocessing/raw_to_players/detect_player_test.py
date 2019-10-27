from detect_player import DetectPlayers
import os

curr_path = os.getcwd()
input_folder = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\testvideo\\test1frames"
output_folder = input_folder + "\\" + "test1_players"
# input_folder = os.path.join(curr_path, 'input_test')
# output_folder = os.path.join(curr_path, 'output_test')
# input_folder = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\raw_frames\\TTY\\raw_tty5\\raw_tty5"
# output_folder = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\raw_frames\\TTY\\raw_tty5\\detected_tty5"
od_model_path = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\tf_object_detection_model\\frozen_inference_graph.pb"

print("Output Folder: " + output_folder)

dp = DetectPlayers(input_folder, output_folder, od_model_path, padding=80)

dp.detect_players()