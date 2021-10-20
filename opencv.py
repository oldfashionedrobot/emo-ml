import io
import os
import cv2
from google.cloud import vision

FILE_NAME = 'NewPicture.jpg'

# Instantiate google vision client
client = vision.ImageAnnotatorClient()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Loop until you hit the Esc key
while True:
    # Capture the current frame
    ret, frame = cap.read()

    # Display the image
    cv2.imshow('Webcam', frame)

    # Detect if the Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        # write the current frame capture to an image file
        cv2.imwrite(FILE_NAME,frame)

        # break out of the while loop
        break


# Release the video capture object
cap.release()
# Close all active windows
cv2.destroyAllWindows()

# Loads the image into memory
with io.open(FILE_NAME, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

print('Labels:')
for label in labels:
    print(label.description)
