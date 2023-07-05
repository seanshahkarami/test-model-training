from torchvision.models import resnet18, resnet34, resnet50, resnet101


def main():
    for modelfn in [resnet18, resnet34, resnet50, resnet101]:
        modelfn(pretrained=True)


if __name__ == "__main__":
    main()
