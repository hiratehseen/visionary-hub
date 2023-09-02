import cv2
from deepface import DeepFace

def detect_emotion(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to RGB (DeepFace uses RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use DeepFace to detect emotions
    results = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)

    # Check if the result is a list (multiple faces) or a dictionary (single face)
    if isinstance(results, list):
        # If there are multiple faces, consider the dominant emotion of the first face
        emotion = results[0]['dominant_emotion']
    elif isinstance(results, dict):
        # If there's a single face, directly extract the dominant emotion
        emotion = results['dominant_emotion']
    else:
        raise ValueError("Invalid results format returned by DeepFace")

    return emotion

if __name__ == "__main__":
    # Path to the input image
    image_path = "image.jpg"

    # Detect emotion in the image
    detected_emotion = detect_emotion(image_path)

    print(f"Detected Emotion: {detected_emotion}")