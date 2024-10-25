



cv2.putText(image, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

# Render detections
mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

cv2.imshow('Mediapipe Feed', image)

if cv2.waitKey(10) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()