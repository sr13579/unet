import torch
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
import matplotlib.pyplot as plt
from train import UNet  

def load_model(model_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, image, device):
    with torch.no_grad():
        image = ToTensor()(image).unsqueeze(0).to(device)
        prediction = model(image)
        return prediction.squeeze(0).cpu()

if __name__ == "__main__":
    test_image_path = 'data/test_images/sample_image.png'
    
    transform = Compose([Resize((128, 128)), ToTensor()])
    test_image = Image.open(test_image_path).convert("L")
    test_image = transform(test_image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("unet_model.pth", device)
    prediction_mask = predict(model, test_image, device)

    plt.imshow(prediction_mask.squeeze(), cmap='gray')
    plt.show()