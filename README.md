# Badminton Pose Analysis using Deep-Learning to improve training
Deep-Learning and AI have found their use in a multitude of areas, and sports is yet another area where they can be super helpful. Badminton is an extremely technical sport and requires great pose and form to prevent injuries. In this work, we present a way to use data from broadcast badminton games to correct posture of amateur players.

## Data Collection
As you can imagine, no public datasets of badminton shots exist, which is why we needed to collect our own data. Here is the procedure we used to collect data:
1. Select top Badminton players to "learn" from: Lee Chong Wei (LCW) of Malaysia, and Tai Tzu Ying (TTY) of Taiwan.
2. Scrape YouTube videos of their broadcast tournament games.
3. Select the types of shots to analyze: Smash, net-drop return, backhand clear and defense.
4. Manually scrub through these videos and extract frames when these shots occur.
5. Process the frames and do the "learning."

![Smash](/images/smash.png)
![Net Drop](/images/drop.png)
![Defense](/images/defense.png)
![Backhand](/images/backhand.png)

## Data Preprocessing
Our aim is to extract pose features from the shots played by these professional players. To be able to do this effectively, we would first need to extract the professional players from these frames. This is done using the following process.
1. Blacken out areas not of interest.
2. Use Google TensorFlow's Object Detection API to detect humans.
3. Pad area around the player of interest and extract the frame.
4. In addition, we store the size and location of the located players which is used later.

## Shot Classification -> Net drop, Smash, Backhand clear and defense
Using ~1400 frames manually collected from over 15 YouTube videos, we built and trained a multi-class classifier that detects what kind of shot the player in the input frame is playing. Our trained model achieved over 93% accuracy in classification. This classification is then passed on to the next stage of the pipeline, which compares the quality of shots played by the test player to the shots played by LCW and TTY. 

![Training Accuracy](/images/accuracy.png)
![Training Loss](/images/loss.png)

## Learned models of shots played by professional Badminton Players
To be able to compare forms of amateur players with professional players, we created learned models of each shot for these two players. For example, we learned the configuration of their skeleton as they play particular types of shots. Consistency is a major factor for success in this sport, and players can keep track of their consistency using this app.

![Pose detection](/images/smashskeleton.png)
![Pose detection](/images/defskeleton.png)

### Features to Compare
1. Net-drop - Lunge distance
2. Smash - Configuration of upper arm, elbow and shoulder
3. Defense - Reachability around the court
4. Backhand - Configuration of lower arm and upper arm

## Evaluation
With the features to compare defined, we apply the same pre-processing and pose detection steps on the input frames, and determine which of the features can be compared.

### Perspective Transform using Homography
Since all tournaments have different camera setups, we cannot simply overlay the pose of players and learn the model. We need to first perform perspective transform to virtually bring the camera scenes together. This is done using the court's boundaries as reference points.

### Bird-eye view
To calculate metrics such as lunge distance and reachability, we need a transformation from 2D camera space to 3D space so that we can measure the distances accurately. To do this, we use the fact that we know the court's dimensions very accurately, and that the entire court is visible in the frame at all times. We then perform a perspective transform to move the viewer's location perpendicularly above the court. This allows us to map the court's dimensions to any euclidean distance on the court. And thus, we can find the lunge distances and reachability of any player.

![Sample Evaluation](/images/radar.png)

This work was developed during CalHacks 6.0 (2019) at UC Berkeley by
* Deepak Talwar
* Seung Won Lee
* Sachin Guruswamy