import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import time

class ImageEmbedder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained = "vggface2", device= device).eval()
    transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    @classmethod
    #PIL image
    def embed(cls, image):
        image_tensor = cls.transforms(image).to(cls.device)
        embeddings = cls.model(image_tensor.unsqueeze(dim = 0))
        return embeddings.cpu().detach().numpy()[0]
    @classmethod
    #PIL images
    def embeds(cls, image_list):
        images_tensor = torch.stack([cls.transforms(image.resize((112, 112))) for image in image_list]).to(cls.device)
        embeddings = cls.model(images_tensor)
        return embeddings.cpu().detach().numpy()

    @classmethod
    #PIL images
    def embeds_to_tensor(cls, image_list):
        images_tensor = torch.stack([cls.transforms(image.resize((112, 112))) for image in image_list]).to(cls.device)
        embeddings = cls.model(images_tensor)
        return embeddings.cpu()
