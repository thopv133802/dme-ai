from PIL import Image
from fastai.vision.augment import Resize, aug_transforms, DataBlock, get_image_files, RandomSplitter, parent_label, \
    ImageBlock, CategoryBlock, Path
from torchvision.transforms import ToPILImage

from utils import FileUtils

class ImageAugmenter:
    size = 112
    data_block = DataBlock(
        blocks = (ImageBlock, CategoryBlock),
        get_items = get_image_files,
        splitter = RandomSplitter(0.0),
        get_y = parent_label,
        item_tfms = Resize(size * 2),
        batch_tfms = aug_transforms(size = size, mult = 0.5)
    )
    to_pillow_image = ToPILImage()
    batch_size = 1
    @classmethod
    def augments(cls, image_folder_path, dest_folder_path, aug_per_image, include_base = False):
        """
        :param include_base: Thêm ảnh ban đầu vào thư mục được aug
        :param aug_per_image: Số ảnh aug ra từ ảnh ban đầu
        :param image_folder_path: đường dẫn tới thư mục lưu các hình ban đầu
            image_folder_path:
                label_1:
                    image_1
                label_2:
                    image_2
        :param dest_folder_path: đường dẫn tới thư mục lưu các hình được aug
            des_folder_path
                label_1
                    image_1
                    augmented_image_1_1
                    augmented_image_1_2
                label_2
                    image_2
                    augmented_image_2_1
                    augmented_image_2_2
        :return: None
        """
        FileUtils.remove_dirs(dest_folder_path)
        FileUtils.make_dirs(dest_folder_path)
        dataloaders = cls.data_block.dataloaders(Path(image_folder_path), bs = cls.batch_size)
        dataloader = dataloaders.train
        vocab = dataloaders.train_ds.vocab
        for aug_index in range(aug_per_image):
            for image_batch, y_batch in dataloader:
                for index, (image, y) in enumerate(zip(image_batch, y_batch)):
                    image = cls.to_pillow_image(image)
                    label = vocab[y]
                    FileUtils.make_dirs(FileUtils.join(dest_folder_path, label))
                    image.save(FileUtils.join(dest_folder_path, label, f"{label}_augmented_{aug_index + 1}.jpg"))
        if include_base:
            for label_name in FileUtils.list_top_folders_names(image_folder_path):
                des_label_folder_path = FileUtils.join(dest_folder_path, label_name)
                FileUtils.make_dirs(des_label_folder_path)
                for file_name in FileUtils.list_top_files_names(FileUtils.join(image_folder_path, label_name)):
                    source_file_path = FileUtils.join(image_folder_path, label_name, file_name)
                    des_file_path = FileUtils.join(des_label_folder_path, file_name)
                    Image.open(FileUtils.join(source_file_path)).resize((cls.size, cls.size)).save(des_file_path)
        # FileUtils.copy(image_folder_path, dest_folder_path)