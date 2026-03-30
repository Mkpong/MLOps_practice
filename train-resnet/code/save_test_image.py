from torchvision import datasets
import os

dataset = datasets.MNIST(root="/home/boan/train-resnet/data", train=False, download=True)

img, label = dataset[0]
img.save("/home/boan/train-resnet/test_mnist.png")

print("saved: /home/boan/train-resnet/test_mnist.png")
print("label:", label)
