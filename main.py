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

    batch_size = 1

    print(
        f"testing {modelfn.__name__} on device {device} with batch size {batch_size}",
        flush=True,
    )

    model = modelfn().to(device)
    model.train()

    optimizer = SGD(model.parameters(), lr=1e-3)

    lossfunc = CrossEntropyLoss()

    # repeat test in case some cache or setup is required
    for i in range(50):
        input = torch.randn((batch_size, 3, 512, 512), device=device)
        target = torch.randint(0, 999, (batch_size,), device=device)

        start = time.time()

        optimizer.zero_grad()

        loss = lossfunc(model(input), target)
        loss.backward()
        optimizer.step()

        duration = time.time() - start
        print("result", modelfn.__name__, device, i, duration, flush=True)


def main():
    for modelfn in [resnet18, resnet34, resnet50, resnet101]:
        test_model_epoch(modelfn)


if __name__ == "__main__":
    main()
