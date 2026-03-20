import cv2

def main():
    # Load the pre-trained Haar Cascade classifier for face detection
    # This cascade detects frontal faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start video capture (0 is usually the default internal webcam)
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit the application.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Convert the frame to grayscale since Haar cascades work best on grayscale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use detectMultiScale3 to get the confidence scores (levelWeights)
        # scaleFactor determines how much the image size is reduced at each image scale
        # minNeighbors determines how many neighbors each candidate rectangle should have to retain it
        faces, rejectLevels, levelWeights = face_cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), outputRejectLevels=True
        )

        # Check if any faces were found
        if len(faces) > 0:
            for i in range(len(faces)):
                x, y, w, h = faces[i]
                
                # Get confidence weight and convert to a heuristic probability between 0 and 100%
                # Haar cascades normally output weights roughly in the 3.0 to 10.0+ range
                weight = float(levelWeights[i])
                probability = min(max((weight / 10.0) * 100, 0), 99.9)
                
                text = f"Face: {probability:.1f}%"

                # Draw a rectangle around each detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Put the probability text on top of the rectangle
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Facial Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
