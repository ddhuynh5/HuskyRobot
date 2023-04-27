import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import Network
from data import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Network().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 150

train_loss = []
train_epoch = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, batch in enumerate(data_loader):
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loss.append(running_loss / len(data_loader))
        train_epoch.append(epoch + i / len(data_loader))
    print('[%d] loss: %f' % (epoch + 1, running_loss / len(data_loader)))

# Plot the training loss
plt.plot(train_epoch, train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

print('Finished Training')

model_path = "model.pt"
torch.save(model.state_dict(), model_path)