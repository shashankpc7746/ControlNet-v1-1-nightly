import cv2
from annotator.openpose import OpenposeDetector

# Initialize the OpenPose detector
openpose = OpenposeDetector()

# Load your character image
image_path = "ControlNet/test_imgs/pose2.png"  # Change this path
image = cv2.imread(image_path)
if image is None:
    raise ValueError("❌ Image not loaded. Please check the path:", image_path)


# Generate the pose
pose_result = openpose(image)
cv2.imshow("Pose", pose_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Save the pose image
output_path = "ControlNet/poses/pose_input.png"
cv2.imwrite(output_path, pose_result)
print(f"✅ Pose image saved to: {output_path}")
