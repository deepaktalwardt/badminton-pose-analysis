import os
import shutil

class RenameFrames():
    def __init__(self, input_folder, output_folder, tournament_name, player_name, frame_type):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.tournament_name = tournament_name
        self.player_name = player_name
        self.frame_type = frame_type

        self.smash_count = 0
        self.drop_count = 0
        self.backhand_count = 0
        self.defense_count = 0
    
    def create_output_folders(self):
        if not os.path.exists(self.output_folder):
            try:
                os.makedirs(self.output_folder)
                os.makedirs(self.output_folder + '/smash')
                os.makedirs(self.output_folder + '/drop')
                os.makedirs(self.output_folder + '/backhand')
                os.makedirs(self.output_folder + '/defense')
                print(self.output_folder + ' and subfolders created')
                return True
            except FileExistsError:
                print(self.output_folder + ' could not be created')
                return False
        else:
            return True

    def rename_frames(self):
        if not self.create_output_folders():
            return
        
        drop_path = self.input_folder + '/drop'
        smash_path = self.input_folder + '/smash'
        backhand_path = self.input_folder + '/backhand'
        defense_path = self.input_folder + '/defense'

        # Drops
        for fn in os.listdir(drop_path):
            if os.path.isfile(os.path.join(drop_path, fn)):
                new_file_name = self.frame_type + '_' + \
                    self.player_name + '_' + \
                    self.tournament_name + '_' + \
                    'drop' + str(self.drop_count + 1) + \
                    '.png'
                shutil.copy(os.path.join(drop_path, fn), os.path.join(self.output_folder, 'drop', new_file_name))
                print('Copied' + fn + ' to output folder as ' + new_file_name)
                self.drop_count += 1
        
        # Backhand
        for fn in os.listdir(backhand_path):
            if os.path.isfile(os.path.join(backhand_path, fn)):
                new_file_name = self.frame_type + '_' + \
                    self.player_name + '_' + \
                    self.tournament_name + '_' + \
                    'backhand' + str(self.backhand_count + 1) + \
                    '.png'
                shutil.copy(os.path.join(backhand_path, fn), os.path.join(self.output_folder, 'backhand', new_file_name))
                print('Copied' + fn + ' to output folder as ' + new_file_name)
                self.backhand_count += 1
        
        # Smash
        for fn in os.listdir(smash_path):
            if os.path.isfile(os.path.join(smash_path, fn)):
                new_file_name = self.frame_type + '_' + \
                    self.player_name + '_' + \
                    self.tournament_name + '_' + \
                    'smash' + str(self.smash_count + 1) + \
                    '.png'
                shutil.copy(os.path.join(smash_path, fn), os.path.join(self.output_folder, 'smash', new_file_name))
                print('Copied' + fn + ' to output folder as ' + new_file_name)
                self.smash_count += 1
        
        # Defense
        for fn in os.listdir(defense_path):
            if os.path.isfile(os.path.join(defense_path, fn)):
                new_file_name = self.frame_type + '_' + \
                    self.player_name + '_' + \
                    self.tournament_name + '_' + \
                    'defense' + str(self.defense_count + 1) + \
                    '.png'
                shutil.copy(os.path.join(defense_path, fn), os.path.join(self.output_folder, 'defense', new_file_name))
                print('Copied' + fn + ' to output folder as ' + new_file_name)
                self.defense_count += 1
        
        print("*********************")
        print("Copied " + str(self.drop_count) + ' drops.')
        print("Copied " + str(self.smash_count) + ' smashes.')
        print("Copied " + str(self.backhand_count) + ' backhands.')
        print("Copied " + str(self.defense_count) + ' defenses.')
        print("*********************")


