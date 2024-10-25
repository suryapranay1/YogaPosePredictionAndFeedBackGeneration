# Yoga Pose Prediction and Feedback Generation

This project uses **MediaPipe** to predict yoga poses and generate feedback to help users improve their postures. The system captures a live video feed, analyzes the user's pose in real time, and provides feedback on adjustments to enhance accuracy and alignment.

## Features
- **Real-time Pose Detection**: Uses MediaPipe for identifying key body points and tracking posture.
- **Yoga Pose Classification**: Classifies poses like Tree, Mountain,Cobra,Corpse and triangle based on keypoint data.
- **Feedback Generation**: Provides corrective feedback based on deviations from the ideal pose to help users improve alignment.
- **Audio Suggestion** : Provide Audio suggestions is there is any misalignment.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/yoga-pose-prediction.git
    cd yoga-pose-prediction
    ```

2. Install dependencies:
    ```
   use pip install 
    ```

3. Run the application only in Pycharm (As we used OpenCv library)
    

## Usage

1. Ensure your camera is functional.
2. Run the application as described above.
3. Follow the on-screen instructions to perform a yoga pose.
4. The app will display real-time feedback to help adjust your posture.


## Output Resources
The `output_resources` folder includes:
- **Pose Model**: Pre-trained model for pose classification.
- **Sample Data**: Example output files showing feedback for various poses.

- ![image](https://github.com/user-attachments/assets/43a76f05-3bf9-448d-a5ed-bf4c6e7163f5)
![image](https://github.com/user-attachments/assets/efd97f9b-d993-4329-9b03-f4959d9e7660)


## Technologies Used
- **MediaPipe**: For real-time pose detection.
- **Python**: Backend for processing and feedback logic.
- **OpenCV**: To capture and display video.

## Future Enhancements
- Improve pose accuracy with more training data.
- Add more poses and personalized feedback.
- Develop a mobile application for broader accessibility.

## Acknowledgments
Special thanks to the MediaPipe team for their amazing tools in real-time machine learning.

