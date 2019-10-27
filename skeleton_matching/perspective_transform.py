import cv2
import numpy as np
import os

class PerspectiveTransform:

    def __init__(self):
        self.court_points = {
            "Malaysia2017": np.array([[421, 287], [858, 287], [1001, 669], [275, 668]]),
            "Japan2017": np.array([[398, 310], [885, 310], [967, 685], [314, 684]]),
            "HongKong2018": np.array([[445, 267], [831, 266], [951, 659], [334, 658]]),
            "Guangzhou2018": np.array([[394, 333], [879, 333], [1054, 661], [230, 661]]),
            "Odense2018": np.array([[459, 277], [841, 277], [1011, 682], [283, 682]]),
            "Fuzhou2017": np.array([[411, 292], [865, 286], [967, 664], [310, 668]])
        }
        self.H_Japan2017_to_Malaysia2017 = self.find_homography_matrix("Japan2017", "Malaysia2017")
        self.H_Fuzhou2017_to_HongKong2018 = self.find_homography_matrix("Fuzhou2017", "HongKong2018")
        self.H_Guangzhou2018_to_HongKong2018 = self.find_homography_matrix("Guangzhou2018", "HongKong2018")
        self.H_Odense2018_to_HongKong2018 = self.find_homography_matrix("Odense2018", "HongKong2018")

        # For bird eye view only
        self.court_width = int(607/2)
        self.court_height = int(1341/2)
    
    def find_homography_matrix(self, key1, key2):
        if key1 in self.court_points.keys() and key2 in self.court_points.keys():
            H, status = cv2.findHomography(self.court_points.get(key1), self.court_points.get(key2))
            return H
        else:
            print("Enter valid tournament names")
    
    def warp_homography_perspective(self, file_path):
        file_name = os.path.basename(file_path)
        fn = file_name.split("_")
        player_name = fn[1]
        tournament_name = fn[2]

        im_orig = cv2.imread(file_path)

        if player_name == "LCW":
            if tournament_name != "Malaysia2017":
                H = self.H_Japan2017_to_Malaysia2017
                warped = cv2.warpPerspective(im_orig, H, (1280, 720))
                cv2.imshow("Source", im_orig)
                cv2.imshow("Warped", warped)
                cv2.waitKey(0)
                return warped
            else:
                return im_orig
        elif player_name == "TTY":
            H = None
            if tournament_name == "HongKong2018":
                return im_orig
            elif tournament_name == "Guangzhou2018":
                H = self.H_Guangzhou2018_to_HongKong2018
            elif tournament_name == "Odense2018":
                H = self.H_Odense2018_to_HongKong2018
            elif tournament_name == "Fuzhou2017":
                H = self.H_Fuzhou2017_to_HongKong2018
            warped = cv2.warpPerspective(im_orig, H, (1280, 720))
            cv2.imshow("Source", im_orig)
            cv2.imshow("Warped", warped)
            cv2.waitKey(0)
            return warped
    
    def bird_eye_view(self, file_path):
        file_name = os.path.basename(file_path)
        fn = file_name.split("_")
        player_name = fn[1]
        tournament_name = fn[2]

        im_orig = cv2.imread(file_path)
        source_points = self.court_points.get(tournament_name).astype('float32')
        dest_points = np.array([[0, 0],
                                [self.court_width - 1, 0],
                                [self.court_width - 1, self.court_height - 1],
                                [0, self.court_height - 1]], dtype='float32')

        M = cv2.getPerspectiveTransform(source_points, dest_points)
        warped = cv2.warpPerspective(im_orig, M, (self.court_width, self.court_height))
        cv2.imshow("Source", im_orig)
        cv2.imshow("Bird eye", warped)
        cv2.waitKey(0)
        return warped

        
if __name__ == '__main__':
    pt = PerspectiveTransform()
    # sample_file_path = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\unlabelled_full_frame_player_renamed\\LCW\\unlabelled_LCW_Japan2017_37.png"
    # sample_file_path = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\raw_frames\\LCW\\LCW4_Japan_2017\\raw_lcw4\\LCW4.00_11_02_18.Still001.png"
    # pt.warp_homography_perspective(sample_file_path)
    
    lcw_dive = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\unlabelled_full_frame_player_renamed\\LCW\\unlabelled_LCW_Japan2017_21.png"
    pt.bird_eye_view(lcw_dive)
