# Badminton Pose Analysis using Deep-Learning to improve training
Deep-Learning and AI have found their use in a multitude of areas, and sports is yet another area where they can be super helpful. Badminton is an extremely technical sport and requires great pose and form to prevent injuries. In this work, we present a way to use data from broadcast badminton games to correct posture of amateur players.

## Data Collection
As you can imagine, no public datasets of badminton shots exist, which is why we needed to collect our own data. Here is the procedure we used to collect data:
1. Select top Badminton players to "learn" from: Lee Chong Wei (LCW) of Malaysia, and Tai Tzu Ying (TTY) of Taiwan.
2. Scrape YouTube videos of their broadcast tournament games.
3. Select the types of shots to analyze: Smash, net-drop return, backhand clear and defense.
4. Manually scrub through these videos and extract frames when these shots occur.
5. Process the frames and do the "learning."

## Data Preprocessing
Our aim is to extract pose features from the shots played by these professional players. To be able to do this effectively, we would first need to extract the professional players from these frames. This is done using the following process.
1. Blacken out areas not of interest.
2. Use Google TensorFlow's Object Detection API to detect humans.
3. Pad area around the player of interest and extract the frame.
4. In addition, we store the size and location of the located players which is used later.

## Shot Classification -> Net drop, Smash, Backhand clear and defense
Using ~1400 frames manually collected from over 15 YouTube videos, we built and trained a multi-class classifier that detects what kind of shot the player in the input frame is playing. Our trained model achieved over 93% accuracy in classification. This classification is then passed on to the next stage of the pipeline, which compares the quality of shots played by the test player to the shots played by LCW and TTY. 

## Pose detection 

## Feature Comparison

## Evaluation




This work was developed during CalHacks 6.0 (2019) at UC Berkeley by
* Deepak Talwar
* Seung Won Lee
* Sachin Guruswamy