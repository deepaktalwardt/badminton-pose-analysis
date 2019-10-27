import cv2
import numpy as np
import os
import json

class PerspectiveTransform:

    def __init__(self):
        self.court_points = {
            "Malaysia2017": np.array([[421, 287], [858, 287], [1001, 669], [275, 668]]),
            "Japan2017": np.array([[398, 310], [885, 310], [967, 685], [314, 684]]),
            "HongKong2018": np.array([[445, 267], [831, 266], [951, 659], [334, 658]]),
            "Guangzhou2018": np.array([[394, 333], [879, 333], [1054, 661], [230, 661]]),
            "Odense2018": np.array([[459, 277], [841, 277], [1011, 682], [283, 682]]),
            "Fuzhou2017": np.array([[411, 292], [865, 286], [967, 664], [310, 668]]),
            "Test1": np.array([[410, 211], [818, 217], [1118, 681], [123, 674]])
        }
        self.H_Japan2017_to_Malaysia2017 = self.find_homography_matrix("Japan2017", "Malaysia2017")
        self.H_Fuzhou2017_to_HongKong2018 = self.find_homography_matrix("Fuzhou2017", "HongKong2018")
        self.H_Guangzhou2018_to_HongKong2018 = self.find_homography_matrix("Guangzhou2018", "HongKong2018")
        self.H_Odense2018_to_HongKong2018 = self.find_homography_matrix("Odense2018", "HongKong2018")
        self.H_Test1_to_Malaysia2017 = self.find_homography_matrix("Test1", "Malaysia2017")

        # For bird eye view only
        self.court_width = int(607/2)
        self.court_height = int(1341/2)
    
    def find_homography_matrix(self, key1, key2):
        if key1 in self.court_points.keys() and key2 in self.court_points.keys():
            H, status = cv2.findHomography(self.court_points.get(key1), self.court_points.get(key2))
            return H
        else:
            print("Enter valid tournament names")
    
    def warp_homography_perspective(self, file_path, show=False):
        file_name = os.path.basename(file_path)
        fn = file_name.split("_")
        player_name = fn[1]
        tournament_name = fn[2]

        im_orig = cv2.imread(file_path)

        if player_name == "LCW":
            if tournament_name != "Malaysia2017":
                H = self.H_Japan2017_to_Malaysia2017
                warped = cv2.warpPerspective(im_orig, H, (1280, 720))
                if show:
                    cv2.imshow("Source", im_orig)
                    cv2.imshow("Warped", warped)
                    cv2.waitKey(0)
                return warped
        elif player_name == "TEST":
            H = self.H_Test1_to_Malaysia2017
            warped = cv2.warpPerspective(im_orig, H, (1280, 720))
            if show:
                cv2.imshow("Source", im_orig)
                cv2.imshow("Warped", warped)
                cv2.waitKey(0)
                
            return warped
                #return im_orig
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
            if show:
                cv2.imshow("Source", im_orig)
                cv2.imshow("Warped", warped)
                cv2.waitKey(0)
            return H, warped
    
    def bird_eye_view(self, file_path, show=False):
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
        if show:
            cv2.imshow("Source", im_orig)
            cv2.imshow("Bird eye", warped)
            cv2.waitKey(0)
        return M, warped
    
    def calculate_distance(self, point1, point2):
        px_distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        return px_distance * (13.410/self.court_height)
    
    def transform_coordinates(self, coordinates, M):
        coordinates = np.array([coordinates[0], coordinates[1], 1])
        transformed = M.dot(coordinates)
        transformed = transformed / transformed[2]
        return (int(transformed[0]), int(transformed[1]))

    def get_lunge_distance(self, file_path, skeleton_info_file, show=False):
        file_name = os.path.basename(file_path)
        fn = file_name.split("_")
        fn = fn[0] + "_" + fn[1] + "_" + fn[2] + "_" + fn[3] + ".png"
        M, warped = self.bird_eye_view(file_path, show=False)
        with open(skeleton_info_file, 'r') as f:
            skeleton_info_dict = json.load(f)
        file_name = os.path.basename(file_path)
        file_name_key = fn[:-4]
        if file_name_key in skeleton_info_dict.keys():
            skeleton_info = skeleton_info_dict[file_name_key]
            dict_to_search = skeleton_info[0]
            if len(skeleton_info) > 1:
                for si in skeleton_info:
                    if len(si) > len(dict_to_search):
                        dict_to_search = si
            if "Rank" in dict_to_search.keys() and "Lank" in dict_to_search.keys():
                r_ank = dict_to_search["Rank"]
                l_ank = dict_to_search["Lank"]
                r_ank = self.transform_coordinates(r_ank, M)
                l_ank = self.transform_coordinates(l_ank, M)
                dist = self.calculate_distance(r_ank, l_ank)

                warped = cv2.circle(warped, l_ank, 20, (0, 0, 255), 3)
                warped = cv2.circle(warped, r_ank, 20, (0, 0, 255), 3)
                warped = cv2.line(warped, l_ank, r_ank, (0,255,255), 5)
                warped = cv2.putText(warped, str(round(dist, 3)) + ' m', (r_ank[0] + 20, r_ank[1] + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

                if show:
                    cv2.imshow("Warped", warped)
                    cv2.waitKey(0)
                
                return dist, warped
        
        print("File path not found in skeleton info dictionary")
        return
    
    def get_extension_distance(self, file_path, skeleton_info_file, show=False):
        file_name = os.path.basename(file_path)
        fn = file_name.split("_")
        fn = fn[0] + "_" + fn[1] + "_" + fn[2] + "_" + fn[3] + ".png"
        M, warped = self.bird_eye_view(file_path, show=False)
        with open(skeleton_info_file, 'r') as f:
            skeleton_info_dict = json.load(f)
        file_name = os.path.basename(file_path)
        file_name_key = fn[:-4]
        if file_name_key in skeleton_info_dict.keys():
            skeleton_info = skeleton_info_dict[file_name_key]
            dict_to_search = skeleton_info[0]
            if len(skeleton_info) > 1:
                for si in skeleton_info:
                    if len(si) > len(dict_to_search):
                        dict_to_search = si
            minX = 10000
            maxX = -1
            min_joint = ""
            max_joint = ""
            for key in dict_to_search.keys():
                value = dict_to_search[key]
                if value[0] < minX:
                    minX = value[0]
                    min_joint = dict_to_search[key]
                elif value[0] > maxX:
                    maxX = value[0]
                    max_joint = dict_to_search[key]
            
            min_joint = self.transform_coordinates(min_joint, M)
            max_joint = self.transform_coordinates(max_joint, M)
            dist = self.calculate_distance(min_joint, max_joint)

            warped = cv2.circle(warped, min_joint, 20, (0, 0, 255), 3)
            warped = cv2.circle(warped, max_joint, 20, (0, 0, 255), 3)
            warped = cv2.line(warped, min_joint, max_joint, (0,255,255), 5)
            warped = cv2.putText(warped, str(round(dist, 3)) + ' m', (max_joint[0] + 20, max_joint[1] + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

            if show:
                cv2.imshow("Warped", warped)
                cv2.waitKey(0)
            
            return dist, warped

    def get_lunge_distance2(self, file_path, skeleton_info_dict, show=False):
        file_name = os.path.basename(file_path)
        # fn = file_name.split("_")
        # fn = fn[0] + "_" + fn[1] + "_" + fn[2] + "_" + fn[3] + ".png"
        M, warped = self.bird_eye_view(file_path, show=False)
        
        #file_name = os.path.basename(file_path)
        file_name = file_name[:-4]
        print(file_name)
        if file_name in skeleton_info_dict.keys():
            skeleton_info = skeleton_info_dict[file_name]
            dict_to_search = skeleton_info[0]
            if len(skeleton_info) > 1:
                for si in skeleton_info:
                    if len(si) > len(dict_to_search):
                        dict_to_search = si
            if "Rank" in dict_to_search.keys() and "Lank" in dict_to_search.keys():
                r_ank = dict_to_search["Rank"]
                l_ank = dict_to_search["Lank"]
                r_ank = self.transform_coordinates(r_ank, M)
                l_ank = self.transform_coordinates(l_ank, M)
                dist = self.calculate_distance(r_ank, l_ank)

                warped = cv2.circle(warped, l_ank, 20, (0, 0, 255), 3)
                warped = cv2.circle(warped, r_ank, 20, (0, 0, 255), 3)
                warped = cv2.line(warped, l_ank, r_ank, (0,255,255), 5)
                warped = cv2.putText(warped, str(round(dist, 3)) + ' m', (r_ank[0] + 20, r_ank[1] + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

                if show:
                    cv2.imshow("Warped", warped)
                    cv2.waitKey(0)
                
                return dist, warped
        
        print("File path not found in skeleton info dictionary")
        return


if __name__ == '__main__':
    pt = PerspectiveTransform()
    # sample_file_path = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\unlabelled_full_frame_player_renamed\\LCW\\unlabelled_LCW_Japan2017_37.png"
    # sample_file_path = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\raw_frames\\LCW\\LCW4_Japan_2017\\raw_lcw4\\LCW4.00_11_02_18.Still001.png"
    # pt.warp_homography_perspective(sample_file_path)
    
    # lcw_dive = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\unlabelled_full_frame_player_renamed\\LCW\\unlabelled_LCW_Japan2017_21.png"
    # pt.bird_eye_view(lcw_dive)

    lcw_drop = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\skeleton_frames\\LCW\\drop\\unlabelled_LCW_Japan2017_76_result.png"
    skeleton_info_file = "C:\\Users\\Deepak Talwar\\Dropbox (Personal)\\SJSU\\Semesters\\Fall2019\\CalHacks\\badminton-pose-analysis\\skeleton_matching\\LCW_drop.json"
    pt.get_lunge_distance(lcw_drop, skeleton_info_file, show=True)
    pt.get_extension_distance(lcw_drop, skeleton_info_file, show=True)