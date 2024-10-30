import timeit
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from utils import save_model
from datasets import *
from SCNet import SCC, Trainer
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=10, help="Window size")
    parser.add_argument("--chunk", type=int, default=1, help="Chunk size")
    parser.add_argument("--device", type=int, default=0, help="Device to be used")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    which_expert = np.array(["A"])
    iteration = 300
    batch = 64
    lr = 0.001
    weight_decay = 1e-4
    momentum = 0.5
    kfold = True
    FREQ = 64

    for we in which_expert:
        x, y, data_name = read_data(freq=FREQ, which_expert=we, window=args.window, chunks=args.chunk, device=device,
                                    print_to_file=True)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1024).split(x, y))
        for fold_number in range(4, 5):
            max_AUC_test = 0
            max_AUC = []
            print(f"Window: {args.window}, Chunk: {args.chunk}, Fold: {fold_number}")
            for i in range(10):
                print("Time: ", i)
                model = SCC(args.window, 18, 128, 2)
                model = model.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
                StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
                train_loader, test_loader = preprocess_data(x, y, folds, fold_number, batch, device)
                trainer = Trainer(model, criterion, optimizer)

                AUCs = ('Epoch\tTime(sec)\tLoss_train\tACC_test\tAUC_test\tPrecision_test\tRecall_test')
                file_AUCs = f"../working/results/expert_{we}_{args.window}sec_{args.chunk}chunk_64Hz_SCC_{fold_number}fold.txt"
                file_model = f"../working/best_models/expert_{we}_{args.window}sec_{args.chunk}chunk_SCC_{fold_number}fold.pt"
                with open(file_AUCs, 'w') as f:
                    f.write(AUCs + '\n')

                    """Start training."""
                    print('Training...')
                    print(AUCs)

                    for epoch in range(1, iteration + 1):
                        start = timeit.default_timer()
                        loss_train = trainer.predict(train_loader).item()
                        ACC_test, AUC_test, precision_test, recall_test = trainer.predict(test_loader,
                                                                                          training=False)
                        if AUC_test > max_AUC_test:
                            # torch.save(model.state_dict, file_model)
                            max_AUC_test = AUC_test
                            max_AUC = [ACC_test, AUC_test, precision_test, recall_test]
                            f.write('\t'.join(map(str, AUCs)) + '\n')
                            epoch_label = epoch
                        StepLR.step()
                        end = timeit.default_timer()
                        time = end - start
                        AUCs = [epoch, time, loss_train, ACC_test, AUC_test, precision_test, recall_test]
                        print('\t'.join(map(str, AUCs)))
                    # print("The best model is epoch", epoch_label)
                    # f.write(f"The best model is epoch:{epoch_label}")
            print(max_AUC)
