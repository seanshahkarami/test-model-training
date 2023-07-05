import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18, resnet34, resnet50, resnet101
import time


def test_model_epoch(modelfn):
    if torch.has_cuda:
        device = "cuda"
    else:
        device = "cpu"

    model = modelfn(pretrained=True).to(device)
    model.train()

    optimizer = SGD(model.parameters(), lr=1e-3)

    lossfunc = CrossEntropyLoss()

    input = torch.randn((4, 3, 512, 512), device=device)
    target = torch.randint(0, 999, (len(input),), device=device)

    start = time.time()

    optimizer.zero_grad()

    loss = lossfunc(model(input), target)
    loss.backward()
    optimizer.step()

    duration = time.time() - start
    print(modelfn.__name__, device, duration, loss.item())


def main():
    for modelfn in [resnet18, resnet34, resnet50, resnet101]:
        test_model_epoch(modelfn)


if __name__ == "__main__":
    main()
