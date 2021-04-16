import os
from shutil import rmtree, copytree, copyfile

import cv2
from PIL import Image


class FileUtils:
    @classmethod
    def list_top_files_names(cls, folder_path):
        for root, folders, files in os.walk(folder_path):
            return files
    @classmethod
    def list_top_folders_names(cls, folder_path):
        for root, folders, files in os.walk(folder_path):
            return folders
    @classmethod
    def join(cls, *p):
        return os.path.join(*p)
    @classmethod
    def make_dirs(cls, path, exists_ok = True):
        os.makedirs(path, exist_ok = exists_ok)

    @classmethod
    def remove_dirs(cls, path):
        if os.path.exists(path):
            rmtree(path)

    @classmethod
    def exists(cls, path):
        return os.path.exists(path)

    @classmethod
    def copy(cls, source, dest, dirs_exist_ok = True):
        copytree(source, dest, dirs_exist_ok = dirs_exist_ok)

    @classmethod
    def copy_file(cls, source_file_path, des_file_path):
        return copyfile(source_file_path, des_file_path)

class GeometryUtils:
    @classmethod
    def calc_box_area(cls, box):
        """
        :param box: (left, upper, right, lower) - tuple.
        :return: area in float
        """
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width * height

class ImageUtils:
    @classmethod
    def cv2_to_pillow_image(cls, cv2_image):
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2_image)