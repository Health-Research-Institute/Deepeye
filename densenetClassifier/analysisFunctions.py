# Import libraries
import torch
import numpy as np

#Create Image Stack
def compute_score_with_logits(logits, labels, device):
    logitsM = torch.max(logits, 1)[1].data # argmax
    if device == "mps":
        one_hots = torch.zeros(*labels.size()).to(device)
    elif device == "cuda":
        one_hots = torch.zeros(*labels.size()).cuda()
    else:
        one_hots = torch.zeros(*labels.size())
    one_hots.scatter_(1, logitsM.view(-1, 1), 1)
    #print("Predictions: \n", one_hots.cpu().numpy())
    scores = (one_hots * labels)

    return scores


def analyseTest(tensorIn):
    prob = tensorIn.cpu().numpy()
    prob = np.round(100*prob)/100
      
    nCl = len(prob[0])
    #initialize probability scorer
    fitN = np.zeros(nCl)
    #find maximum probability index 
    mpi =np.argmax(prob, axis=-1) #8-element vector w highest prob

    for j in range(nCl):
        fitN[j] = sum(np.equal(mpi, j))

    return fitN