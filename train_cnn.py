import torch.optim as optim
from functions import overlapScore
from torch.utils.data import DataLoader
from cnn_model import *
from training_dataset import *

def train_model(net, dataloader, batchSize, lr_rate):
    criterion = nn.MSELoss()
    optimization = optim.SGD(net.parameters(), lr=lr_rate)

    # Check if CUDA is available
    if torch.cuda.is_available():
        net.cuda()

    for epoch in range(30):

        for i, data in enumerate(dataloader):
            optimization.zero_grad()

            inputs, labels = data

            # Move inputs and labels to the GPU if available
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            inputs, labels = inputs.view(batchSize, 1, 100, 100), labels.view(batchSize, 4)

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimization.step()

            pbox = outputs.detach().cpu().numpy()  # Move back to CPU for numpy operations
            gbox = labels.detach().cpu().numpy()
            score, _ = overlapScore(pbox, gbox)

            print('[epoch %5d, step: %d, loss: %f, Average Score = %f' % (epoch+1, i+1, loss.item(), score/batchSize))

    print('Finish Training')

if __name__ == '__main__':
    # Hyperparameters
    learning_rate = 0.1
    batch = 100
    no_of_workers = 2
    shuffle = True

    trainingdataset = training_dataset()
    dataLoader = DataLoader(
        dataset=trainingdataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=no_of_workers
    )

    model = cnn_model()
    model.train()

    train_model(model, dataLoader, batch, learning_rate)
    torch.save(model.state_dict(), './Model/cnn_model.pth')
