from PIL import Image

import torchvision.transforms as transforms

norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img = Image.open('/usr/local/data/images/faces/Processed-VN_Celeb/VN_CELEB/images/person_0/3863.png').convert('RGB')
img = img.resize((256, 256), Image.BILINEAR)
img = norm(img)
print(img.shape)
