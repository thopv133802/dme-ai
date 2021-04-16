import time

import numpy
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from utils import GeometryUtils

class FaceExtracter:
    model = MTCNN(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    @classmethod
    def extracts(cls, image: Image, return_bounding_boxes = False):
        faces_bounding_boxes, confidents = cls.model.detect(image)
        if faces_bounding_boxes is None or len(faces_bounding_boxes) == 0:
            return False
        faces = []
        for face_bounding_box, confident in zip(faces_bounding_boxes, confidents):
            face = image.crop(face_bounding_box)
            faces.append(face)
        return faces if not return_bounding_boxes else faces, faces_bounding_boxes
    @classmethod
    def extract(cls, image: Image):
        faces_bounding_boxes, confidents = cls.model.detect(image)
        if len(faces_bounding_boxes) == 0:
            return False
        if len(faces_bounding_boxes) == 1:
            largest_face_bounding_box = faces_bounding_boxes[0]
        else:
            faces_areas = [GeometryUtils.calc_box_area(face_bounding_box) for face_bounding_box in faces_bounding_boxes]
            largest_face_index = numpy.argmax(faces_areas)
            largest_face_bounding_box = faces_bounding_boxes[largest_face_index]
        return image.crop(largest_face_bounding_box)
