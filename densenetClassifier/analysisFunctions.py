# Import libraries
import torch

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