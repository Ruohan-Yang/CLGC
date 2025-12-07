import os
from tqdm import trange
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from src.PGD import PGD

def train_phase(model, train_loader, valid_loader, args, log, best_valid_dir, log_mode):

    best_valid_result = 0
    best_model_epoch = 0
    patience = args.patience
    current_patience = 0
    history_loss = []
    bak_loss = 0
    history_min_loss = 10000
    pgd = PGD(model, 'cnn', epsilon=args.epsilon, alpha=args.alpha)
    K = args.disturb_t
    criterion = nn.CrossEntropyLoss()

    print('Training...')
    if log_mode == 'support':
        log.write('Training...\n')

    for epoch in trange(args.epochs, desc="Training"):
        if args.ifDecay:
            p = epoch / (args.epochs - 1)
            learning_rate = args.learning_rate / pow((1 + 10 * p), 0.75)
        else:
            learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

        model.train()
        whole_loss_vec = []
        train_acc_vec = []
        for data in train_loader:
            network_labels, left_nodes, right_nodes, link_labels = data
            network_labels = Variable(network_labels).cuda()
            left_nodes = Variable(left_nodes).cuda()
            right_nodes = Variable(right_nodes).cuda()
            link_labels = Variable(link_labels).cuda()

            prediction_outs, discriminant_outs = model(network_labels, left_nodes, right_nodes)

            prediction_loss = criterion(prediction_outs, link_labels)
            discriminant_loss = criterion(discriminant_outs, network_labels)
            whole_loss = prediction_loss + discriminant_loss
            whole_loss_vec.append(whole_loss.cpu().detach().numpy())

            _, argmax = torch.max(prediction_outs, 1)
            batch_acc = (argmax == link_labels).float().mean()
            train_acc_vec.append(batch_acc.item())

            whole_loss.backward()

            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))

                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()

                _, adv_outs = model(network_labels, left_nodes, right_nodes)
                adv_loss = criterion(adv_outs, network_labels)
                adv_loss.backward()
            pgd.restore_origin_grad()
            pgd.restore()
            optimizer.step()
            optimizer.zero_grad()

        whole_loss = np.mean(whole_loss_vec)
        history_loss.append(whole_loss)
        # train_acc = np.mean(train_acc_vec)

        # write_infor = "Epoch:[{}/{}], lr:{:.4f}, Loss:{:.4f}, Train acc:{:.4f}".format(epoch+1, args.epochs, learning_rate, whole_loss, train_acc)
        # print(write_infor)
        # log.write(write_infor)

        model.eval()
        eval_results = model.metrics_eval(valid_loader)
        # write_infor = ', '.join([f"{k}: {v:.4f}" for k, v in eval_results.items()])
        # print("Valid " + write_infor)
        # log.write('\n' + "Valid " + write_infor + '\n')

        # Update best_valid_metric and early stopping counter
        # Save the best model
        valid_result = eval_results[args.best_metric]
        if valid_result > best_valid_result:
            print(f"Current best -> epoch: {epoch}, {args.best_metric} value: {valid_result:.4f}; Previous best -> epoch: {best_model_epoch}, {args.best_metric} value: {best_valid_result:.4f}")
            best_valid_result = valid_result
            bak_loss = whole_loss
            current_patience = 0
            torch.save(model.state_dict(), best_valid_dir)
            best_model_epoch = epoch
        else:
            current_patience += 1

        # Check if early stopping conditions are met
        history_min_loss = min(whole_loss, history_min_loss)  # if plateau
        if args.EarlyStop and current_patience >= patience and whole_loss > history_min_loss + 0.01:
            print(f"Early stopping triggered. Best epoch: {best_model_epoch}, best {args.best_metric}: {best_valid_result:.4f}")
            print(f'epoch {best_model_epoch} loss: {bak_loss}. Latest epoch {epoch} loss: {whole_loss}. history_min_loss: {history_min_loss}. history_loss: {history_loss}')
            break

    write_infor = 'Best Epoch:[{}/{}]'.format(best_model_epoch+1, args.epochs)
    print(write_infor)
    if log_mode == 'support':
        log.write(write_infor + '\n')
    return best_valid_dir

def test_phase(model, best_valid_dir, test_loader, log):
    # Load the best model for testing
    print('Load best model ' + best_valid_dir + ' ... ')
    model.load_state_dict(torch.load(best_valid_dir))
    model.eval()
    test_results = model.metrics_eval(test_loader, mode='test')
    write_infor = ', '.join([f"{k}: {v:.4f}" for k, v in test_results.items()])
    print("Test " + write_infor)
    log.write("Test " + write_infor + '\n')
    return test_results

def run_model(train_loader, valid_loader, test_loader, model, args, log_mode='support'):
    log = open(args.log, 'a', encoding='utf-8')
    if log_mode == 'support':
        best_valid_dir = 'save/' + args.dataset + '_best_model.pth'
    else:
        best_valid_dir = 'save/' + 'best_model.pth'
    if args.onlyTest and os.path.exists(best_valid_dir):
        print('onlyTest, best_model exists')
        log.write('onlyTest\n')
    else:
        best_valid_dir = train_phase(model, train_loader, valid_loader, args, log, best_valid_dir, log_mode)
    test_results = test_phase(model, best_valid_dir, test_loader, log)
    log.close()
    return test_results
