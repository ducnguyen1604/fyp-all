import torch
from PIL import Image
from torchvision.transforms import ToTensor
from models.backbone import get_resnet50_backbone
from models.cssa import CSSA
from configs.config import config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_inference(image_path):
    backbone = get_resnet50_backbone(pretrained=False).to(DEVICE)
    cssa_module = CSSA(in_channels=2048, 
                       reduction_factor=config['reduction_factor'],
                       cssa_thresh=config['cssa_switching_thresh']).to(DEVICE)

    checkpoint = torch.load(config['checkpoint_path'])
    backbone.load_state_dict(checkpoint['backbone'])
    cssa_module.load_state_dict(checkpoint['cssa'])

    backbone.eval()
    cssa_module.eval()

    img = Image.open(image_path).convert('RGB')
    img_tensor = ToTensor()(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = backbone(img_tensor)
        output = cssa_module(features)

    return output.cpu()

if __name__ == '__main__':
    img_path = 'path/to/test_image.jpg'
    prediction = run_inference(img_path)
    print("Inference complete. Output shape:", prediction.shape)
