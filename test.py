import io
import os
import cv2
from google.cloud import vision
from PIL import Image, ImageDraw
import glob


def detect_face(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of Face objects with information about the picture.
    """
    client = vision.ImageAnnotatorClient()

    content = face_file.read()
    image = vision.Image(content=content)

    return client.face_detection(
        image=image, max_results=max_results)


def highlight_faces(image, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.

    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    # Sepecify the font-family and the font-size
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')

    for face in faces:
        print('anger: {}'.format(
            likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(
            likelihood_name[face.surprise_likelihood]))
        print('sorrow: {}'.format(
            likelihood_name[face.sorrow_likelihood]))

        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')

        # Place the confidence value/score of the detected faces above the
        # detection box in the output image
        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FFFF00')

        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y + 20),
                  'joy: {}'.format(likelihood_name[face.joy_likelihood]),
                  fill='#FFFFFF')

        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y + 40),
                  'surprise: {}'.format(
                      likelihood_name[face.surprise_likelihood]),
                  fill='#FFFFFF')

        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y + 60),
                  'anger: {}'.format(
                      likelihood_name[face.anger_likelihood]),
                  fill='#FFFFFF')

        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y + 80),
                  'sorrow: {}'.format(
                      likelihood_name[face.sorrow_likelihood]),
                  fill='#FFFFFF')

    im.save(output_filename)




count = 0
max = 10000
for filename in glob.glob('fairface-img-margin025-trainval/val/*.jpg'):
    with io.open(filename, 'rb') as image_file:
        max_results = 4
        results = detect_face(image_file, max_results)
        faces = results.face_annotations

        print(faces)

        print(image_file)

        print('Writing to file {}'.format('output/test-img-' + str(count) + '.jpg'))
        # Reset the file pointer, so we can read the file again
        image_file.seek(0)
        highlight_faces(image_file, faces, 'output/test-img-' + str(count) + '.jpg')

    print("done an image, incrementing count")
    count += 1
    if(count > max):
        break
