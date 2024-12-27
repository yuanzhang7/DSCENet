from utils.utils import *
import os
from utils.dataset_generic import save_splits

from Models.model_DSCE import DSCE
from Models.model_mil import MIL_fc, MIL_fc_mc_earlyclinic

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type == 'mil':
        if args.n_classes > 2:
            model = MIL_fc_mc_earlyclinic(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    elif args.model_type == 'DSCE':

        model_dict = {
            "clinic_factor": args.clinic_factor,
            "n_classes": args.n_classes,
            "fusion": args.fusion,
        }
        model = DSCE(**model_dict)


    else:
        print('no model_type available!')


    if torch.cuda.is_available():
        model = model.cuda()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 30, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')
    val_auc_score = 0.5

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,args.clinic_factor)
        stop, val_auc= validate(cur, epoch, model, val_loader, args.n_classes,
            early_stopping, writer, loss_fn, args.results_dir,args.clinic_factor)
        ##根据valid数据集存储best model-----
        if val_auc_score < val_auc:
            val_auc_score = val_auc
            torch.save(model.state_dict(),
                       os.path.join(args.results_dir, "s_{}_Maximum_valid_auc_checkpoint.pt".format(cur)))
            best_auc_model = model
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    val_results_dict, val_error, val_auc, _= summary(model, val_loader, args.n_classes,args.clinic_factor)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes,args.clinic_factor)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    ###最佳模型
    _, best_auc_val_error, best_auc_val_auc, _ = summary(best_auc_model, val_loader, args.n_classes,args.clinic_factor)
    print('Best_auc_model: Val error: {:.4f}, ROC AUC: {:.4f}'.format(best_auc_val_error, best_auc_val_auc))
    best_auc_results_dict, best_auc_test_error, best_auc_test_auc, best_auc_acc_logger = summary(best_auc_model, test_loader, args.n_classes,args.clinic_factor)
    print('Best_auc_model: Test error: {:.4f}, ROC AUC: {:.4f}'.format(best_auc_test_error, best_auc_test_auc))


    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    # return results_dict, test_auc, val_auc, 1-test_error, 1-val_error
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error ,best_auc_results_dict, best_auc_test_auc, best_auc_val_auc, 1-best_auc_test_error, 1-best_auc_val_error


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None,clinic_factor=10):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    print('\n')

    for batch_idx, (data, label, clinic, coords) in enumerate(loader):
        if clinic_factor == 10:
            clinic_data1 = [
                torch.tensor(df[["age", "gender", "hb", "wbc", "rbc", "hct", "plt", "jak2", "mpl", "calr"]].values) for
                df in clinic]
        elif clinic_factor == 7:
            clinic_data1 = [torch.tensor(df[["age", "gender", "hb", "wbc", "rbc", "hct", "plt"]].values) for df in
                            clinic]
        elif clinic_factor == 3:
            clinic_data1 = [torch.tensor(df[["jak2", "mpl", "calr"]].values) for df in clinic]
        elif clinic_factor == 0:
            clinic_data1 = [torch.zeros(0)]
        else:
            print('not know how many clinic data')
        clinic_data = clinic_data1[0].to(torch.float32)
        coords = torch.cat(coords, dim=0)
        data, label, clinic_data, coords= data.to(device), label.to(device), clinic_data.to(device),coords.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data, clinic_data, coords)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(),
                                                                           data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None,clinic_factor=10):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, clinic,coords) in enumerate(loader):
            if clinic_factor == 10:
                clinic_data1 = [
                    torch.tensor(df[["age", "gender", "hb", "wbc", "rbc", "hct", "plt", "jak2", "mpl", "calr"]].values)
                    for df in clinic]
            elif clinic_factor == 7:
                clinic_data1 = [torch.tensor(df[["age", "gender", "hb", "wbc", "rbc", "hct", "plt"]].values) for df in
                                clinic]
            elif clinic_factor == 3:
                clinic_data1 = [torch.tensor(df[["jak2", "mpl", "calr"]].values) for df in clinic]
            elif clinic_factor == 0:
                clinic_data1 = [torch.zeros(0)]

            else:
                print('not know how many clinic data')

            clinic_data = clinic_data1[0].to(torch.float32)
            coords = torch.cat(coords, dim=0)
            data, label, clinic_data, coords= data.to(device, non_blocking=True), label.to(device, non_blocking=True), clinic_data.to(device, non_blocking=True), coords.to(device, non_blocking=True)
            logits, Y_prob, Y_hat, _, _ = model(data, clinic_data,coords)
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        # 将 NaN 替换为 0
        if np.isnan(prob).any():
            print('original prob:', prob)
            nan_indices = np.isnan(prob)# 先检查是否有 NaN 值
            prob[nan_indices] = 0.25  # 将 NaN 替换为 0
            print('prob1:', prob)
            # prob = prob / np.sum(prob, axis=1, keepdims=True)
            # print('prob2:', prob)

        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            # return True
            return True, auc
    return False, auc

def summary(model, loader, n_classes,clinic_factor):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, clinic,coords) in enumerate(loader):
        if clinic_factor == 10:
            clinic_data1 = [
                torch.tensor(df[["age", "gender", "hb", "wbc", "rbc", "hct", "plt", "jak2", "mpl", "calr"]].values)
                for df in clinic]
        elif clinic_factor == 7:
            clinic_data1 = [torch.tensor(df[["age", "gender", "hb", "wbc", "rbc", "hct", "plt"]].values) for df in
                            clinic]
        elif clinic_factor == 3:
            clinic_data1 = [torch.tensor(df[["jak2", "mpl", "calr"]].values) for df in clinic]
        elif clinic_factor == 0:
            clinic_data1 = [torch.zeros(0)]

        else:
            print('not know how many clinic data')

        clinic_data =  clinic_data1[0].to(torch.float32)
        coords = torch.cat(coords, dim=0)
        data, label, clinic_data,coords= data.to(device), label.to(device), clinic_data.to(device),coords.to(device)
    # for batch_idx, (data, label) in enumerate(loader):
    #     data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            # logits, Y_prob, Y_hat, _, _ = model(data)
            logits, Y_prob, Y_hat, _, _ = model(data,clinic_data,coords)
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs,'prob_class':Y_hat, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
