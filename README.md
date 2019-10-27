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
Our aim is to extract pose features from the shots played by these professional players. To be able to do this effectively, we would first need to extract the professional players from these 




This work was developed during CalHacks 6.0 (2019) at UC Berkeley by
* Deepak Talwar
* Seung Won Lee
* Sachin Guruswamy