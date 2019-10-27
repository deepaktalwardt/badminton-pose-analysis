import os
import glob
import csv
import shutil

class RenameUnlabelledFrames():
    def __init__(self, input_folder, output_folder, csv_file, tournament_name, player_name, frame_type, start_index=0):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.tournament_name = tournament_name
        self.player_name = player_name
        self.frame_type = frame_type
        self.csv_file = csv_file
        self.output_csv_file = None
        self.start_index = start_index

    def _create_output_folder(self):
        self.output_csv_file = self.output_folder + '\\' + 'output_detection_info.csv'
        # if os.path.isfile(output_csv_file):
        #     os.remove(output_csv_file)
        # try:
        #     os.makedirs(output_csv_file)
        #     self.output_csv_file = output_csv_file
        # except:
        #     print("ERROR: Couldn't create CSV file")
        #     return False
        if not os.path.exists(self.output_folder):
            try:
                os.makedirs(self.output_folder)
                print(self.output_folder + ' and output CSV file created')
                return True
            except:
                print("Output folder could not be created")
                return False;
        return True

    def get_detection_info(self, file_name):
        with open(self.csv_file, 'r') as fr:
            csv_reader = csv.reader(fr, delimiter=',')
            for row in csv_reader:
                if row[0] == file_name:
                    return [row[1], row[2], row[3], row[4]]
    
    def rename_unlabelled_frames(self):
        if not self._create_output_folder():
            return False
        frame_list = glob.glob(self.input_folder + '/*.png')
        count = start_index
        for frame in frame_list:
            old_file_name = os.path.basename(frame)
            new_file_name = self.frame_type + '_' + \
                self.player_name + '_' + \
                self.tournament_name + '_' + \
                str(count + 1) + \
                '.png'
            detection_info = self.get_detection_info(old_file_name)
            shutil.copy(os.path.join(self.input_folder, frame), os.path.join(self.output_folder, new_file_name))
            with open(self.output_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([new_file_name, old_file_name] + detection_info)
            count += 1
            print("Renamed " + old_file_name + " to " + new_file_name)


if __name__ == "__main__":
    input_folder = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\testvideo\\test1frames\\test1_players"
    output_folder = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\unlabelled_test_player_renamed"
    input_csv_file = input_folder + "\\detection_info.csv"
    tournament_name = "Test1"
    player_name = "TEST"
    frame_type = "unlabelled"
    start_index = 0
    ruf = RenameUnlabelledFrames(input_folder, output_folder, input_csv_file, tournament_name, player_name, frame_type)
    ruf.rename_unlabelled_frames()