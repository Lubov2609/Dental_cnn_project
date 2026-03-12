import torch


def predict(model, image, device):

    model.eval()

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

    return output.cpu().numpy()