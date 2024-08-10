"""MultiPhysNet Trainer."""
import os
from collections import OrderedDict
import sys
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.MultiPhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
import pdb
import csv

class MultiPhysNetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.model_name = config.MODEL.NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        
        self.min_valid_loss = None
        self.best_epoch = 0
        self.task = config.TASK
        self.dataset_type = config.DATASET_TYPE # face finger not task
        self.train_state = config.TRAIN.DATA.INFO.STATE
        self.valid_state = config.VALID.DATA.INFO.STATE
        self.test_state = config.TEST.DATA.INFO.STATE
        self.lr = config.TRAIN.LR
        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.MultiPhysNet.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("MultiPhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []  
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            running_loss_bvp = 0.0
            running_loss_spo2 = 0.0
            train_loss = []
            train_loss_bvp = []
            train_loss_spo2 = []

            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=120)
            

            
            for idx, batch in enumerate(tbar):
                
                # print(f"batch: {batch}")
                
                tbar.set_description("Train epoch %s" % epoch)
                loss_bvp = torch.tensor(0.0)  # Initialize to avoid UnboundLocalError
                loss_spo2 = torch.tensor(0.0)  # Initialize to avoid UnboundLocalError
                # self.dataset_type
                if self.dataset_type != "both":
                    if self.task == "bvp":
                        rPPG, _,= self.model(
                            batch[0].to(torch.float32).to(self.device))
                        BVP_label = batch[1].to(torch.float32).to(self.device)
                        rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                        loss = self.loss_model(rPPG, BVP_label)
                        running_loss_bvp += loss.item()
                    elif self.task == "spo2":
                        _, spo2_pred,= self.model(
                            batch[0].to(torch.float32).to(self.device))
                        spo2_label = batch[2].to(torch.float32).to(self.device).squeeze(-1)
                        # print(f"spo2_pred.shape: {spo2_pred.shape}, spo2_label: {{spo2_label.shape}}")
                        # loss = torch.nn.MSELoss()(spo2_pred, spo2_label) pass
                        # loss = torch.nn.SmoothL1Loss()(spo2_pred, spo2_label) # 149 124 118 115 105 98.6 99.8 100 102 92.9 92.4 12  9.5

                        # loss = torch.nn.L1Loss()(spo2_pred, spo2_label) 
                        loss = torch.nn.MSELoss()(spo2_pred, spo2_label)*(100-spo2_label.mean())**2

                        running_loss_spo2 += loss.item()
                    elif self.task == "both":

                        rPPG, spo2_pred = self.model(
                            batch[0].to(torch.float32).to(self.device))
                        BVP_label = batch[1].to(torch.float32).to(self.device)
                        spo2_label = batch[2].to(torch.float32).to(self.device).squeeze(-1)
                        rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)

                        loss_bvp = self.loss_model(rPPG, BVP_label)
                        # print(f"spo2_pred.shape: {spo2_pred.shape}, spo2_label: {{spo2_label.shape}}")
                        # loss_spo2 = torch.nn.MSELoss()(spo2_pred, spo2_label)
                        # loss_spo2 = torch.nn.L1Loss()(spo2_pred, spo2_label)
                        loss_spo2 = torch.nn.MSELoss()(spo2_pred, spo2_label)*(100-spo2_label.mean())**2

                        loss = 100 * loss_bvp + loss_spo2
                        running_loss_bvp += loss_bvp.item()
                        running_loss_spo2 += loss_spo2.item()
                        
                    else:
                        raise ValueError(f"Unknown task: {self.task}")
                
                    
                    
                else:  # both face and finger
                    face_data = batch[0].to(torch.float32).to(self.device)
                    finger_data = batch[1].to(torch.float32).to(self.device)
                    
                    if self.task == "bvp":
                        rPPG, _= self.model(face_data, finger_data)
                        BVP_label = batch[2].to(torch.float32).to(self.device)
                        rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                        # print(f"rPPG.shape: {rPPG.shape}, BVP_label.shape: {BVP_label.shape}")
                        loss = self.loss_model(rPPG, BVP_label)
                        running_loss_bvp += loss.item()
                    elif self.task == "spo2":
                        _, spo2_pred = self.model(face_data, finger_data)
                        spo2_label = batch[3].to(torch.float32).to(self.device).squeeze(-1)
                        # loss = torch.nn.L1Loss()(spo2_pred, spo2_label)
                        loss = torch.nn.MSELoss()(spo2_pred, spo2_label)*(100-spo2_label.mean())**2
                        
                        running_loss_spo2 += loss.item()
                    elif self.task == "both":
                        rPPG, spo2_pred = self.model(face_data, finger_data)
                        BVP_label = batch[2].to(torch.float32).to(self.device)
                        spo2_label = batch[3].to(torch.float32).to(self.device).squeeze(-1)
                        rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                        
                        loss_bvp = self.loss_model(rPPG, BVP_label)
                        
                        loss_spo2 = torch.nn.MSELoss()(spo2_pred, spo2_label)*(100-spo2_label.mean())**2
                        
                        loss = 100 * loss_bvp + loss_spo2 
                        running_loss_bvp += loss_bvp.item()
                        running_loss_spo2 += loss_spo2.item()
                    else:
                        raise ValueError(f"Unknown task: {self.task}")
                loss.backward()
                running_loss += loss.item()
                train_loss.append(loss.item())
                if self.task in ["bvp", "both"]:
                    train_loss_bvp.append(running_loss_bvp)
                if self.task in ["spo2", "both"]:
                    train_loss_spo2.append(running_loss_spo2)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()
                lrs.append(self.scheduler.get_last_lr()[0])  
                self.optimizer.zero_grad()
            tbar.set_postfix(loss=loss.item(), loss_bvp=running_loss_bvp, loss_spo2=running_loss_spo2)
            print(f"train loss: {np.mean(train_loss)}")
            # if self.task  in ["bvp", "both"]:
            #     mean_training_losses.append(np.mean(train_loss_bvp))
            # if self.task in ["spo2", "both"]:
            #     mean_training_losses.append(np.mean(train_loss_spo2))
            if self.task == "bvp" :
                mean_training_losses.append(np.mean(train_loss_bvp))
            if self.task == "spo2" :
                mean_training_losses.append(np.mean(train_loss_spo2))
            if self.task == "both" :
                mean_training_losses.append(np.mean(train_loss))
            
            
                
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                with open ('/data2/lk/rppg-toolbox/loss.csv', 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    data_to_add = [
                        epoch+1, np.mean(train_loss), valid_loss
                    ]
                    csv_writer.writerow(data_to_add)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        valid_loss_bvp = 0.0
        valid_loss_spo2 = 0.0
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                if self.dataset_type != "both": # face or finger
                
                
                    if self.task == "bvp":
                        BVP_label = valid_batch[1].to(torch.float32).to(self.device)
                        rPPG, _ = self.model(
                            valid_batch[0].to(torch.float32).to(self.device))
                        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)  # normalize
                        loss = self.loss_model(rPPG, BVP_label)
                        valid_loss_bvp += loss.item()
                    elif self.task == "spo2":
                        spo2_label = valid_batch[2].to(torch.float32).to(self.device).squeeze(-1)
                        _, spo2_pred = self.model(
                            valid_batch[0].to(torch.float32).to(self.device))
                        # loss = torch.nn.L1Loss()(spo2_pred, spo2_label)
                        loss = torch.nn.MSELoss()(spo2_pred, spo2_label)*(100-spo2_label.mean())**2
                        
                        valid_loss_spo2 += loss.item()
                    # do both task
                    else:        
                        data = valid_batch[0].to(torch.float32).to(self.device)
                        rPPG, spo2_pred = self.model(data)
                        BVP_label = valid_batch[1].to(torch.float32).to(self.device)
                        spo2_label = valid_batch[2].to(torch.float32).to(self.device).squeeze(-1)
                        rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                        loss_bvp = self.loss_model(rPPG, BVP_label)
                        loss_spo2 = torch.nn.MSELoss()(spo2_pred, spo2_label)*(100-spo2_label.mean())**2
                            
                            # loss_spo2 = torch.nn.L1Loss()(spo2_pred, spo2_label)
                        loss = 100 * loss_bvp + loss_spo2
                        valid_loss_bvp += loss_bvp.item()
                        valid_loss_spo2 += loss_spo2.item()
                        
                
                else: # both
                        
                        face_data = valid_batch[0].to(torch.float32).to(self.device)
                        finger_data = valid_batch[1].to(torch.float32).to(self.device)
                        if self.task == "both":
                            rPPG, spo2_pred = self.model(face_data, finger_data)
                                # rppg, spo2 = self.model(face_data, finger_data)
                            BVP_label = valid_batch[2].to(torch.float32).to(self.device)
                            spo2_label = valid_batch[3].to(torch.float32).to(self.device).squeeze(-1)
                            rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                            BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                            loss_bvp = self.loss_model(rPPG, BVP_label)
                                
                            loss_spo2 = torch.nn.MSELoss()(spo2_pred, spo2_label)*(100-spo2_label.mean())**2
                                
                                # loss_spo2 = torch.nn.L1Loss()(spo2_pred, spo2_label)
                            loss = 100 * loss_bvp + loss_spo2
                            valid_loss_bvp += loss_bvp.item()
                            valid_loss_spo2 += loss_spo2.item() 
                        elif self.task == "bvp":        
                            rPPG, _= self.model(face_data, finger_data)
                            BVP_label = valid_batch[2].to(torch.float32).to(self.device)
                            rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                            BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                            # print(f"rPPG.shape: {rPPG.shape}, BVP_label.shape: {BVP_label.shape}")
                            loss = self.loss_model(rPPG, BVP_label)
                            valid_loss_bvp += loss.item()
                        elif self.task == "spo2":        
                            _, spo2_pred = self.model(face_data, finger_data)
                            spo2_label = valid_batch[3].to(torch.float32).to(self.device).squeeze(-1)
                            # loss = torch.nn.L1Loss()(spo2_pred, spo2_label)
                            loss = torch.nn.MSELoss()(spo2_pred, spo2_label)*(100-spo2_label.mean())**2
                            valid_loss_spo2 += loss.item()
                        
    
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item(), loss_bvp=valid_loss_bvp, loss_spo2=valid_loss_spo2)

            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)


    def test(self, data_loader):
        """Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print("\n===Testing===")
        rppg_predictions = dict()
        spo2_predictions = dict()
        rppg_labels = dict()
        spo2_labels = dict()
        print(f"dataset_type: {self.dataset_type}")
        # define column names
        header = [
            'V_TYPE', 'TASK', 'LR','Epoch Number', 'HR_MAE', 'HR_MAE_STD', 'HR_RMSE', 'HR_RMSE_STD',
            'HR_MAPE', 'HR_MAPE_STD', 'HR_Pearson', 'HR_Pearson_STD', 'HR_SNR','HR_SNR_STD',
            'SPO2_MAE', 'SPO2_MAE_STD', 'SPO2_RMSE', 'SPO2_RMSE_STD', 'SPO2_MAPE',
            'SPO2_MAPE_STD', 'SPO2_Pearson', 'SPO2_Pearson_STD', 'SPO2_SNR','SPO2_SNR_STD',
            'Model', 'train_state', 'valid_state','test_state'
        ] 
        
        

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                
                if self.dataset_type == "both":
                    # For multi-modal input, combine face and finger data
                    face_data = test_batch[0].to(self.config.DEVICE)
                    finger_data = test_batch[1].to(self.config.DEVICE)

                    # print(f"test_batch_shape: {test_batch}")
                    # combined_data = torch.cat((face_data, finger_data), dim=1)
                    
                    rppg_label = test_batch[2].to(self.config.DEVICE)
                    spo2_label = test_batch[3].to(self.config.DEVICE)
                    

                    
                    pred_ppg_test, pred_spo2_test = self.model(face_data, finger_data)
                else:
                    data, rppg_label, spo2_label = test_batch[0].to(
                        self.config.DEVICE), test_batch[1].to(self.config.DEVICE), test_batch[2].to(self.config.DEVICE)
                    pred_ppg_test, pred_spo2_test= self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    rppg_label = rppg_label.cpu()
                    spo2_label = spo2_label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                    pred_spo2_test = pred_spo2_test.cpu()
                
                if self.dataset_type == "both":
                    
                    for idx in range(batch_size):
                        subj_index = test_batch[4][idx]
                        sort_index = int(test_batch[5][idx])
                        if subj_index not in rppg_predictions:
                            rppg_predictions[subj_index] = dict()
                            rppg_labels[subj_index] = dict()
                        rppg_predictions[subj_index][sort_index] = pred_ppg_test[idx]
                        rppg_labels[subj_index][sort_index] = rppg_label[idx]
                            
                    for idx in range(batch_size):
                        subj_index = test_batch[4][idx]
                        sort_index = int(test_batch[5][idx])
                        if subj_index not in spo2_predictions:
                            spo2_predictions[subj_index] = dict()
                            spo2_labels[subj_index] = dict()
                        spo2_predictions[subj_index][sort_index] = pred_spo2_test[idx]
                        spo2_labels[subj_index][sort_index] = spo2_label[idx]
                else:
                    # print(f"test_batch_shape: {len(test_batch)}")  
                    # print(f"test_batch: {test_batch}")  
                    for idx in range(batch_size):
                        subj_index = test_batch[3][idx]
                        sort_index = int(test_batch[4][idx])
                        if subj_index not in rppg_predictions:
                            rppg_predictions[subj_index] = dict()
                            rppg_labels[subj_index] = dict()
                        rppg_predictions[subj_index][sort_index] = pred_ppg_test[idx]
                        rppg_labels[subj_index][sort_index] = rppg_label[idx]
                            
                    for idx in range(batch_size):
                        subj_index = test_batch[3][idx]
                        sort_index = int(test_batch[4][idx])
                        if subj_index not in spo2_predictions:
                            spo2_predictions[subj_index] = dict()
                            spo2_labels[subj_index] = dict()
                        spo2_predictions[subj_index][sort_index] = pred_spo2_test[idx]
                        spo2_labels[subj_index][sort_index] = spo2_label[idx]
                    

        print('')
        file_exists = os.path.isfile('/data2/lk/rppg-toolbox/result2.csv')
        with open('/data2/lk/rppg-toolbox/result2.csv', 'a', newline='') as csvfile:
            # inference => How to be more Lupin
            #epoch_num = int(self.config.INFERENCE.MODEL_PATH.split('/')[-1].split('.')[0].split('_')[-1][5:]) + 1
            epoch_num = self.max_epoch_num #train
            csv_writer = csv.writer(csvfile)

            if not file_exists:
                csv_writer.writerow(header)
            if self.task == "bvp":
                result = calculate_metrics(rppg_predictions, rppg_labels, self.config, "rppg")
                metrics = result["metrics"]
                # MAE RMSE MAPE Pearson SNR
                HR_MAE, HR_MAE_STD = metrics.get("FFT_MAE", (None, None))
                HR_RMSE, HR_RMSE_STD = metrics.get("FFT_RMSE", (None, None))
                HR_MAPE, HR_MAPE_STD = metrics.get("FFT_MAPE", (None, None))
                HR_Pearson, HR_Pearson_STD = metrics.get("FFT_Pearson", (None, None))
                HR_SNR, HR_SNR_STD = metrics.get("FFT_SNR", (None, None)) if "FFT_SNR" in metrics else (None, None)

                
                data_to_add = [
                    self.dataset_type, self.task, self.lr,epoch_num, HR_MAE, HR_MAE_STD, HR_RMSE, HR_RMSE_STD,
                    HR_MAPE, HR_MAPE_STD, HR_Pearson, HR_Pearson_STD, HR_SNR, HR_SNR_STD,
                    "/", "/", "/", "/", "/", "/",
                    "/", "/", "/","/",
                    self.model_name, self.train_state, self.valid_state, self.test_state
                ]
            elif self.task == "spo2":
                result = calculate_metrics(spo2_predictions, spo2_labels, self.config, "spo2")
                metrics = result["metrics"]
                # MAE RMSE MAPE Pearson
                SPO2_MAE, SPO2_MAE_STD = metrics.get("FFT_MAE", (None, None))
                SPO2_RMSE, SPO2_RMSE_STD = metrics.get("FFT_RMSE", (None, None))
                SPO2_MAPE, SPO2_MAPE_STD = metrics.get("FFT_MAPE", (None, None))
                SPO2_Pearson, SPO2_Pearson_STD = metrics.get("FFT_Pearson", (None, None))          
                data_to_add = [
                    self.dataset_type, self.task, self.lr, epoch_num, "/", "/", "/", "/",
                    "/", "/", "/", "/", "/", "/",
                    SPO2_MAE, SPO2_MAE_STD, SPO2_RMSE, SPO2_RMSE_STD, SPO2_MAPE, SPO2_MAPE_STD,
                    SPO2_Pearson, SPO2_Pearson_STD, "/", "/",
                    self.model_name, self.train_state, self.valid_state, self.test_state
                ]       
            
            
            elif self.task == "both":
                result_rppg = calculate_metrics(rppg_predictions, rppg_labels, self.config, "rppg")
                result_spo2 = calculate_metrics(spo2_predictions, spo2_labels, self.config, "spo2")
                metrics_rppg = result_rppg["metrics"]
                HR_MAE, HR_MAE_STD = metrics_rppg.get("FFT_MAE", (None, None))
                HR_RMSE, HR_RMSE_STD = metrics_rppg.get("FFT_RMSE", (None, None))
                HR_MAPE, HR_MAPE_STD = metrics_rppg.get("FFT_MAPE", (None, None))
                HR_Pearson, HR_Pearson_STD = metrics_rppg.get("FFT_Pearson", (None, None))
                HR_SNR, HR_SNR_STD = metrics_rppg.get("FFT_SNR", (None, None))
                metrics_spo2 = result_spo2["metrics"]
                SPO2_MAE, SPO2_MAE_STD = metrics_spo2.get("FFT_MAE", (None, None))
                SPO2_RMSE, SPO2_RMSE_STD = metrics_spo2.get("FFT_RMSE", (None, None))
                SPO2_MAPE, SPO2_MAPE_STD = metrics_spo2.get("FFT_MAPE", (None, None))
                SPO2_Pearson, SPO2_Pearson_STD = metrics_spo2.get("FFT_Pearson", (None, None))
                data_to_add = [
                    self.dataset_type, self.task, self.lr,epoch_num, HR_MAE, HR_MAE_STD, HR_RMSE, HR_RMSE_STD,
                    HR_MAPE, HR_MAPE_STD, HR_Pearson, HR_Pearson_STD, HR_SNR, HR_SNR_STD,
                    SPO2_MAE, SPO2_MAE_STD, SPO2_RMSE, SPO2_RMSE_STD, SPO2_MAPE, SPO2_MAPE_STD,
                    SPO2_Pearson, SPO2_Pearson_STD, "/", "/",
                    self.model_name, self.train_state, self.valid_state,self.test_state
                ]         
                
                # self.config.INFERENCE.MODEL_PATH
                if self.config.INFERENCE.MODEL_PATH:
                    epoch_number = self.extract_epoch_from_path(self.config.INFERENCE.MODEL_PATH)
                    data_to_add_hr_spo2_MAE = [
                        epoch_number, HR_MAE, SPO2_MAE
                    ]          
        
        # write data
            if self.config.TOOLBOX_MODE != "only_test":
                csv_writer.writerow(data_to_add)
            else:
                # only_test  hr_spo2_MAE
                with open("/data2/lk/rppg-toolbox/MAE.csv", 'a', newline='') as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow(data_to_add_hr_spo2_MAE)
                
        if self.config.TEST.OUTPUT_SAVE_DIR:  # saving test outputs 
            self.save_test_outputs(rppg_predictions, rppg_labels, self.config)
            self.save_test_outputs(spo2_predictions, spo2_labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def save_test_outputs(self, predictions, labels, config):
        if not os.path.exists(config.TEST.OUTPUT_SAVE_DIR):
            os.makedirs(config.TEST.OUTPUT_SAVE_DIR)
        output_file = os.path.join(config.TEST.OUTPUT_SAVE_DIR, f"{self.model_file_name}_test_outputs.npz")
        np.savez(output_file, predictions=predictions, labels=labels)
        print(f"Saved test outputs to: {output_file}")
    # when inference
    def extract_epoch_from_path(self, model_path):
        print(model_path)
        parts = model_path.split('/')
        for part in parts:
            if 'Epoch' in part:

                a = part.find("Epoch")
                b = part.find(".pth")
                epoch_str = part[a+5:b]
                return int(epoch_str)+1
        raise ValueError("The model path does not contain an epoch number.")
