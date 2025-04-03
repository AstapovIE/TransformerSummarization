import torch

from prepare_data import prepare_data
from model.Transformer import Transformer
from Training import LabelSmoothingLoss, NoamOpt, fit
from hometasks_functions import task1, task3

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print("device:", DEVICE)


train_iter, test_iter, word_field = prepare_data(DEVICE)


##########################################################  MODEL  #############################################################

model = Transformer(vocab=word_field.vocab, d_model=768, device=DEVICE, embed='bert', blocks_count=4).to(DEVICE)
pad_idx = word_field.vocab.stoi['<pad>']
criterion_LB = LabelSmoothingLoss(vocab_size=len(word_field.vocab), padding_idx=pad_idx, smoothing=0.1).to(DEVICE)
optimizer = NoamOpt(model)

##########################################################  TRAIN  #############################################################

# trained_model, train_loss, val_loss = fit(model, criterion_LB, optimizer, train_iter, epochs_count=2, val_iter=test_iter,
#                                           device=DEVICE, use_wandb=False, use_TF=None)
#
##########################################################  SAVE  #############################################################
# torch.save(trained_model.state_dict(), "none_2.pth")

model.load_state_dict(torch.load("none_29.pth", map_location=DEVICE))
model.eval()

##########################################################  TASK  #############################################################
task1(model, test_iter, 5, DEVICE, "data/demo_result_none_29")


