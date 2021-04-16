import numpy
from PIL import Image
from tqdm import tqdm

from face_classifier import FaceClassifier
from face_extracter import FaceExtracter
from image_augmenter import ImageAugmenter
from image_embedder import ImageEmbedder
from utils import FileUtils, ImageUtils
import cv2
import time


def extract_faces(portrait_folder_path="portrait", face_folder_path="face"):
    print("extract_faces", "start")
    FileUtils.remove_dirs(face_folder_path)
    FileUtils.make_dirs(face_folder_path)
    for label in tqdm(FileUtils.list_top_folders_names(portrait_folder_path)):
        for file_name in FileUtils.list_top_files_names(FileUtils.join(portrait_folder_path, label)):
            portrait = Image.open(FileUtils.join(portrait_folder_path, label, file_name))
            face = FaceExtracter.extract(portrait)
            if not face:
                print(f"Không tìm thấy khuôn mặt nào trong chân dung {file_name} của {label}")
                continue
            label_face_folder_path = FileUtils.join(face_folder_path, label)
            face_file_path = FileUtils.join(label_face_folder_path, file_name)
            FileUtils.make_dirs(label_face_folder_path)
            face.save(face_file_path)
    print("extract_faces", "completed")


def augment_faces(face_folder_path="face", augmented_folder_path="face_augmented", aug_per_image = 8):
    print("augment_faces", "start")

    FileUtils.remove_dirs(augmented_folder_path)
    FileUtils.make_dirs(augmented_folder_path)

    ImageAugmenter.augments(face_folder_path, augmented_folder_path, aug_per_image = aug_per_image)

    print("augment_faces", "completed")


def train_classifier(data_folder_path="face_augmented"):
    print("train_classifier", "start")
    X = []
    y = []
    print("train_classifier", "embed_faces", "start")
    for label in tqdm(FileUtils.list_top_folders_names(data_folder_path)):
        faces = []
        for file_name in FileUtils.list_top_files_names(FileUtils.join(data_folder_path, label)):
            face = Image.open(FileUtils.join(data_folder_path, label, file_name))
            faces.append(face)
        embeds = ImageEmbedder.embeds(faces)
        X += [embed for embed in embeds]
        y += [label] * len(faces)
    print("train_classifier", "embed_faces", "completed")

    print("train_classifier", "fit_classifier", "start")
    FaceClassifier.fit(numpy.array(X), numpy.array(y))
    print("train_classifier", "fit_classifier", "completed")


def start_recognize_faces_stream():
    capture = cv2.VideoCapture(0)
    while True:
        _, frame = capture.read()
        pil_frame = ImageUtils.cv2_to_pillow_image(frame)
        time_label = []
        current_time = time.time()
        faces, bounding_boxes = FaceExtracter.extracts(pil_frame, return_bounding_boxes=True)
        extract_time = time.time() - current_time
        time_label.append("extract_time: {}".format(extract_time))

        if faces:
            for bounding_box in bounding_boxes:
                left, top, right, bottom = bounding_box.astype(numpy.int)
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255))

            current_time = time.time()
            faces_embeddings = ImageEmbedder.embeds(faces)
            embed_time = time.time() - current_time
            time_label.append("embed_time: {}".format(embed_time))

            current_time = time.time()
            labels, confidents = FaceClassifier.classifies(faces_embeddings)
            classify_time = time.time() - current_time
            time_label.append("classify_time: {}".format(classify_time))

            for (label, confident, bounding_box) in zip(labels, confidents, bounding_boxes):
                left, top, right, bottom = bounding_box.astype(numpy.int)
                cv2.putText(frame,  f"{label}[{confident}]", (left, max(top - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        cv2.putText(frame, ",".join(time_label), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # extract_faces("portrait", "face")
    # augment_faces("face", "face_augmented")
    # train_classifier("face_augmented")
    start_recognize_faces_stream()



