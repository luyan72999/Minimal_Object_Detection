import numpy as np
import pandas as pd
from functions import overlapScore
import torch
from cnn_model import cnn_model

testX = pd.read_csv('Dataset/testData.csv', sep=',', header=None)
groundTruth = pd.read_csv('Dataset/ground-truth-test.csv', sep=',', header=None)

testX = np.asanyarray(testX)
groundTruth = np.asarray(groundTruth)
groundTruth = np.asarray(groundTruth)


model = cnn_model()
model.eval()
model.load_state_dict(torch.load('Model/cnn_model.pth'))


output = model(torch.Tensor(np.reshape(testX, (len(testX),1,100,100))) / 255.0) * 100

output = output.detach().numpy()
output = output.astype(int)

score, _ = overlapScore(output, groundTruth)
score /= len(testX)
print('Test Average overlap score : %f' % score)

np.savetxt('Results/test-result.csv', output, delimiter=',')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
for i in range(len(testX)):
    img = testX[i].reshape(100, 100)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    pred_box = output[i]
    pred_rect = patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2], pred_box[3], linewidth=1, edgecolor='r', facecolor='none', label='Predicted')
    ax.add_patch(pred_rect)

    gt_box = groundTruth[i]
    gt_rect = patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2], gt_box[3], linewidth=1, edgecolor='g', facecolor='none', label='Ground Truth')
    ax.add_patch(gt_rect)

    plt.legend()
    plt.show()

