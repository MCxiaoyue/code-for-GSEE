import os
import torch
from models import models as networks
from models.models_HiFi import Generator as model_HiFi
from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss, save_checkpoint
from modules import mel2wav_vocoder, perform_STT
from utils import data_denorm, word_index
import torch.nn as nn
import torch.nn.functional as F
from NeuroTalkDataset import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate
import json
import argparse
import wavio
from torch.utils.tensorboard import SummaryWriter

    
def train(args, train_loader, models, criterions, optimizers, epoch, trainValid=True):
    '''
    :param args: general arguments
    :param train_loader: loaded for training/validation/test dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: losses
    '''
    (optimizer_g, optimizer_d) = optimizers
    
    # switch to train mode
    assert type(models) == tuple, "More than two models should be inputed (generator and discriminator)"

    epoch_loss_g = []
    epoch_loss_d = []
    
    epoch_acc_g = []
    epoch_acc_d = []
    
    epoch_loss_g_ns = []
    epoch_loss_d_ns = []
    
    epoch_acc_g_ns = []
    epoch_acc_d_ns = []

    total_batches = len(train_loader)

    for i, (input, target, target_cl, voice, data_info) in enumerate(train_loader):    

        print("\rBatch [%5d / %5d]"%(i,total_batches), sep=' ', end='', flush=True)
        
        input = input.cuda()
        target = target.cuda()
        target_cl = target_cl.cuda()
        voice = torch.squeeze(voice,dim=-1).cuda()
        labels = torch.argmax(target_cl,dim=1) 
        
        # extract unseen
        idx_unseen=[]
        idx_seen=[]
        for j in range(len(labels)):
            if args.classname[labels[j]] == args.unseen:
                idx_unseen.append(j)
            else:
                idx_seen.append(j)
        
        input_ns = input[idx_unseen]
        target_ns = target[idx_unseen]
        target_cl_ns = target_cl[idx_unseen]
        voice_ns = voice[idx_unseen]
        labels_ns = labels[idx_unseen]
        data_info_ns = [data_info[0][idx_unseen],data_info[1][idx_unseen]]
        
        input = input[idx_seen]
        target = target[idx_seen]
        target_cl = target_cl[idx_seen]
        voice = voice[idx_seen]
        labels = labels[idx_seen]
        data_info = [data_info[0][idx_seen],data_info[1][idx_seen]]
        
        # # need to remove
        # models = (model_g, model_d, vocoder, model_STT, decoder_STT)
        # criterions = (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER)
        # trainValid = True
        
        # general training
        if len(input) != 0:
            # train generator
            mel_out, e_loss_g, e_acc_g = train_G(args,
                                                 input, target, voice, labels,
                                                 models, criterions, optimizer_g,
                                                 data_info,
                                                 trainValid)
            epoch_loss_g.append(e_loss_g)
            epoch_acc_g.append(e_acc_g)

            # train discriminator
            e_loss_d, e_acc_d = train_D(args,
                                        mel_out, target, target_cl, labels,
                                        models, criterions, optimizer_d,
                                        trainValid)
            epoch_loss_d.append(e_loss_d)
            epoch_acc_d.append(e_acc_d)

        # Unseen words training
        if len(input_ns) != 0 :
            # Unseen train generator
            mel_out_ns, e_loss_g_ns, e_acc_g_ns = train_G(args,
                                                          input_ns, target_ns, voice_ns, labels_ns,
                                                          models, criterions, optimizer_g,
                                                          data_info_ns,
                                                          False)
            epoch_loss_g_ns.append(e_loss_g_ns)
            epoch_acc_g_ns.append(e_acc_g_ns)

            # Unseen train discriminator
            e_loss_d_ns, e_acc_d_ns = train_D(args,
                                              mel_out_ns, target_ns, target_cl_ns, labels_ns,
                                              models, criterions, optimizer_d,
                                              False)
            epoch_loss_d_ns.append(e_loss_d_ns)
            epoch_acc_d_ns.append(e_acc_d_ns)

    epoch_loss_g = np.array(epoch_loss_g)
    epoch_acc_g = np.array(epoch_acc_g)
    epoch_loss_d = np.array(epoch_loss_d)
    epoch_acc_d = np.array(epoch_acc_d)
    
    epoch_loss_g_ns = np.array(epoch_loss_g_ns)
    epoch_acc_g_ns = np.array(epoch_acc_g_ns)
    epoch_loss_d_ns = np.array(epoch_loss_d_ns)
    epoch_acc_d_ns = np.array(epoch_acc_d_ns)
    
    
    args.loss_g = sum(epoch_loss_g[:,0]) / len(epoch_loss_g[:,0])
    args.loss_g_recon = sum(epoch_loss_g[:,1]) / len(epoch_loss_g[:,1])
    args.loss_g_valid = sum(epoch_loss_g[:,2]) / len(epoch_loss_g[:,2])
    args.loss_g_ctc = sum(epoch_loss_g[:,3]) / len(epoch_loss_g[:,3])
    args.acc_g_valid = sum(epoch_acc_g[:,0]) / len(epoch_acc_g[:,0])
    args.cer_gt = sum(epoch_acc_g[:,1]) / len(epoch_acc_g[:,1])
    args.cer_recon = sum(epoch_acc_g[:,2]) / len(epoch_acc_g[:,2])
    
    # args.loss_d = sum(epoch_loss_d[:,0]) / len(epoch_loss_d[:,0])
    # args.loss_d_valid = sum(epoch_loss_d[:,1]) / len(epoch_loss_d[:,1])
    # args.loss_d_cl = sum(epoch_loss_d[:,2]) / len(epoch_loss_d[:,2])

    # 【修改后】增加 real 和 fake 的平均值计算
    # 注意：同样需要应用之前的 "safe_mean" 逻辑以防空数组报错
    def safe_mean(arr, col_idx=0):
        if arr.size == 0: return 0.0
        if arr.ndim == 1: return arr.mean()
        return arr[:, col_idx].mean()

    args.loss_d = safe_mean(epoch_loss_d, 0)
    args.loss_d_valid = safe_mean(epoch_loss_d, 1)
    args.loss_d_cl = safe_mean(epoch_loss_d, 2)

    # 新增这两行
    args.loss_d_real = safe_mean(epoch_loss_d, 3)
    args.loss_d_fake = safe_mean(epoch_loss_d, 4)

    args.acc_d_real = sum(epoch_acc_d[:,0]) / len(epoch_acc_d[:,0])
    args.acc_d_fake = sum(epoch_acc_d[:,1]) / len(epoch_acc_d[:,1])
    args.acc_cl_real = sum(epoch_acc_d[:,2]) / len(epoch_acc_d[:,2])
    args.acc_cl_fake = sum(epoch_acc_d[:,3]) / len(epoch_acc_d[:,3])
    
    # Unseen
    # args.loss_g_ns = sum(epoch_loss_g_ns[:,0]) / len(epoch_loss_g_ns[:,0])
    # args.loss_g_recon_ns = sum(epoch_loss_g_ns[:,1]) / len(epoch_loss_g_ns[:,1])
    # args.loss_g_valid_ns = sum(epoch_loss_g_ns[:,2]) / len(epoch_loss_g_ns[:,2])
    # args.loss_g_ctc_ns = sum(epoch_loss_g_ns[:,3]) / len(epoch_loss_g_ns[:,3])
    # args.acc_g_valid_ns = sum(epoch_acc_g_ns[:,0]) / len(epoch_acc_g_ns[:,0])
    # args.cer_gt_ns = sum(epoch_acc_g_ns[:,1]) / len(epoch_acc_g_ns[:,1])
    # args.cer_recon_ns = sum(epoch_acc_g_ns[:,2]) / len(epoch_acc_g_ns[:,2])
    #
    # args.loss_d_ns = sum(epoch_loss_d_ns[:,0]) / len(epoch_loss_d_ns[:,0])
    # args.loss_d_valid_ns = sum(epoch_loss_d_ns[:,1]) / len(epoch_loss_d_ns[:,1])
    # args.loss_d_cl_ns = sum(epoch_loss_d_ns[:,2]) / len(epoch_loss_d_ns[:,2])
    # args.acc_d_real_ns = sum(epoch_acc_d_ns[:,0]) / len(epoch_acc_d_ns[:,0])
    # args.acc_d_fake_ns = sum(epoch_acc_d_ns[:,1]) / len(epoch_acc_d_ns[:,1])
    # args.acc_cl_real_ns = sum(epoch_acc_d_ns[:,2]) / len(epoch_acc_d_ns[:,2])
    # args.acc_cl_fake_ns = sum(epoch_acc_d_ns[:,3]) / len(epoch_acc_d_ns[:,3])
    # Helper function to safely calculate mean for potentially 1D or 2D arrays
    def safe_mean(arr, col_idx=0):
        if arr.size == 0:
            return 0.0  # 如果数组为空，返回 0
        if arr.ndim == 1:
            return arr.mean()  # 如果是一维，直接求平均
        return arr[:, col_idx].mean()  # 如果是二维，取指定列求平均

    # Unseen (安全计算，防止 IndexError)
    args.loss_g_ns = safe_mean(epoch_loss_g_ns, 0)
    args.loss_g_recon_ns = safe_mean(epoch_loss_g_ns, 1)
    args.loss_g_valid_ns = safe_mean(epoch_loss_g_ns, 2)
    args.loss_g_ctc_ns = safe_mean(epoch_loss_g_ns, 3)

    args.acc_g_valid_ns = safe_mean(epoch_acc_g_ns, 0)
    args.cer_gt_ns = safe_mean(epoch_acc_g_ns, 1)
    args.cer_recon_ns = safe_mean(epoch_acc_g_ns, 2)

    args.loss_d_ns = safe_mean(epoch_loss_d_ns, 0)
    args.loss_d_valid_ns = safe_mean(epoch_loss_d_ns, 1)
    args.loss_d_cl_ns = safe_mean(epoch_loss_d_ns, 2)

    args.acc_d_real_ns = safe_mean(epoch_acc_d_ns, 0)
    args.acc_d_fake_ns = safe_mean(epoch_acc_d_ns, 1)
    args.acc_cl_real_ns = safe_mean(epoch_acc_d_ns, 2)
    args.acc_cl_fake_ns = safe_mean(epoch_acc_d_ns, 3)

    # tensorboard
    # 【修改点】只有当 args 中存在 writer 属性时才执行记录，防止 eval.py 报错
    if hasattr(args, 'writer') and args.writer is not None:
        if trainValid:
            tag = 'train'
        else:
            tag = 'valid'

        args.writer.add_scalar("Loss_G/{}".format(tag), args.loss_g, epoch)
        args.writer.add_scalar("CER/{}".format(tag), args.cer_recon, epoch)

        # 注意：原代码中这里引用了 args.acc_g_cl，如果上面没定义可能会报错，建议确认是否应为 args.acc_g_valid
        # 根据你的上一版代码，这里应该是 args.acc_g_valid
        if hasattr(args, 'acc_g_valid'):
            args.writer.add_scalar("ACC_G/{}".format(tag), args.acc_g_valid, epoch)

        args.writer.add_scalar("Loss_G_recon/{}".format(tag), args.loss_g_recon, epoch)
        args.writer.add_scalar("Loss_G_valid/{}".format(tag), args.loss_g_valid, epoch)
        args.writer.add_scalar("Loss_G_ctc/{}".format(tag), args.loss_g_ctc, epoch)

        # 确保这些变量已定义（参考之前的修复，如果 unseen 数据为空，它们现在是 0.0）
        if hasattr(args, 'loss_d_real'):
            args.writer.add_scalar("Loss_D_real/{}".format(tag), args.loss_d_real, epoch)
        if hasattr(args, 'loss_d_fake'):
            args.writer.add_scalar("Loss_D_fake/{}".format(tag), args.loss_d_fake, epoch)

        if hasattr(args, 'loss_g_ns'):
            args.writer.add_scalar("Loss_G_unseen/{}".format(tag), args.loss_g_ns, epoch)
        if hasattr(args, 'cer_recon_ns'):
            args.writer.add_scalar("CER_unseen/{}".format(tag), args.cer_recon_ns, epoch)

        # 刷新写入
        args.writer.flush()
    else:
        # 如果没有 writer (例如在 eval.py 中)，可以选择打印日志或者什么都不做
        print("\n[Info] TensorBoard writer not found, skipping logging.")

    print('\n[%3d/%3d] G_valid: %.8f D_R: %.8f D_F: %.8f / CER-gt: %.8f CER-recon: %.8f / g-RMSE: %.8f g-lossValid: %.8f g-lossCTC: %.8f'
          % (i, total_batches, 
             args.acc_g_valid, args.acc_d_real, args.acc_d_fake, 
             args.cer_gt, args.cer_recon, 
             args.loss_g_recon, args.loss_g_valid, args.loss_g_ctc))
        
        
    return (args.loss_g, args.loss_g_recon, args.loss_g_valid, args.loss_g_ctc, args.acc_g_valid, args.cer_gt, args.cer_recon, args.loss_d, args.acc_d_real, args.acc_d_fake, args.acc_d_fake)


def train_G(args, input, target, voice, labels, models, criterions, optimizer_g, data_info, trainValid):

    (model_g, model_d, vocoder, model_STT, decoder_STT) = models
    (criterion_recon, criterion_ctc, criterion_adv, _, CER) =  criterions
    
    if trainValid:
        model_g.train()
        model_d.train()
        vocoder.train()
        model_STT.train()
    else:
        model_g.eval()
        model_d.eval()
        vocoder.eval()
        model_STT.eval()
    
    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(input), 1), dtype=torch.float32).cuda()
    
    ###############################
    # Train Generator
    ###############################
    
    if trainValid:
        for p in model_g.parameters():
            p.requires_grad_(True)   # unfreeze G
        for p in model_d.parameters():
            p.requires_grad_(False)  # freeze D
        for p in vocoder.parameters():
            p.requires_grad_(False)  # freeze vocoder
        for p in model_STT.parameters():
            p.requires_grad_(False)  # freeze model_STT
            
        # set zero grad    
        optimizer_g.zero_grad()
        
        # Run Generator
        output = model_g(input)
    else:
        with torch.no_grad():
            # run generator
            output = model_g(input)
    
    # DTW
    mel_out = DTW_align(output, target)
    
    # Run Discriminator
    g_valid, _ = model_d(mel_out)
    
    # generator loss
    loss_recon = criterion_recon(mel_out, target)
    
    # GAN loss
    loss_valid = criterion_adv(g_valid, valid)
    
    # accuracy    args.l_g = h_g.l_g
    acc_g_valid = (g_valid.round() == valid).float().mean()
    
    ###############################
    # Loss from Vocoder - STT
    ###############################
    # out_DTW
    target_denorm = data_denorm(target, data_info[0], data_info[1])
    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
    
    gt_label=[]
    gt_label_idx=[]
    gt_length=[]
    for j in range(len(target)):
        gt_label.append(args.word_label[labels[j].item()])
        gt_label_idx.append(args.word_index[labels[j].item()])
        gt_length.append(args.word_length[labels[j].item()])
    gt_label_idx = torch.tensor(np.array(gt_label_idx),dtype=torch.int64)
    gt_length = torch.tensor(gt_length,dtype=torch.int64)
    
    # target
    ##### HiFi-GAN
    wav_target = vocoder(target_denorm)
    wav_target = torch.reshape(wav_target, (len(wav_target),wav_target.shape[-1]))
    
    #### resampling
    wav_target = torchaudio.functional.resample(wav_target, args.sample_rate_mel, args.sample_rate_STT)
    if wav_target.shape[1] !=  voice.shape[1]:
        p = voice.shape[1] - wav_target.shape[1]
        p_s = p//2
        p_e = p-p_s
        wav_target = F.pad(wav_target, (p_s,p_e))

    # recon
    ##### HiFi-GAN
    wav_recon = vocoder(output_denorm)
    wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))

    #### resampling
    # wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)
    # if wav_recon.shape[1] !=  voice.shape[1]:
    #     p = voice.shape[1] - wav_recon.shape[1]
    #     p_s = p//2
    #     p_e = p-p_s
    #     wav_recon = F.pad(wav_recon, (p_s,p_e))

    #### resampling for RECON
    wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)

    # === 🔴 新增：对 Ground Truth (voice) 也进行重采样 ===
    # voice 当前是 22050 Hz (来自 Dataset)，必须转到 16000 Hz
    # 假设 args.sample_rate_mel 代表原始音频采样率 (22050)
    voice = torchaudio.functional.resample(voice, args.sample_rate_mel, args.sample_rate_STT)
    # ===================================================

    # Padding 对齐 (确保两者长度一致)
    # 注意：重采样后长度可能会微调，需要重新检查对齐
    if wav_recon.shape[1] != voice.shape[1]:
        # 以较长的为基准，或者取最小公倍数，这里简单处理为对齐到 voice
        target_len = max(wav_recon.shape[1], voice.shape[1])

        # Pad wav_recon
        if wav_recon.shape[1] < target_len:
            p = target_len - wav_recon.shape[1]
            p_s = p // 2
            p_e = p - p_s
            wav_recon = F.pad(wav_recon, (p_s, p_e))

        # Pad voice
        if voice.shape[1] < target_len:
            p = target_len - voice.shape[1]
            p_s = p // 2
            p_e = p - p_s
            voice = F.pad(voice, (p_s, p_e))

    ##### STT Wav2Vec 2.0
    emission_gt, _ = model_STT(voice)
    emission_recon, _ = model_STT(wav_recon)
   
    # CTC loss
    input_lengths = torch.full(size=(emission_gt.size(dim=0),), fill_value=emission_gt.size(dim=1), dtype=torch.long)
    emission_recon_ = emission_recon.log_softmax(2)
    loss_ctc = criterion_ctc(emission_recon_.transpose(0, 1), gt_label_idx, input_lengths, gt_length) 
    
    # total generator loss
    loss_g = args.l_g[0] * loss_recon + args.l_g[1] * loss_valid + args.l_g[2] * loss_ctc

    # decoder STT
    transcript_gt = []
    transcript_recon = []

    for j in range(len(voice)):
        transcript = decoder_STT(emission_gt[j])   
        transcript_gt.append(transcript)
            
        transcript = decoder_STT(emission_recon[j])
        transcript_recon.append(transcript)

    cer_gt = CER(transcript_gt, gt_label)
    cer_recon = CER(transcript_recon, gt_label)

    if trainValid:
        loss_g.backward()
        # # === 🔴 新增：梯度裁剪 (防止 CTC 或其他 Loss 导致梯度爆炸) ===
        # torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
        # # =============================================================
        optimizer_g.step()
    
    e_loss_g = (loss_g.item(), loss_recon.item(), loss_valid.item(), loss_ctc.item())
    e_acc_g = (acc_g_valid.item(), cer_gt.item(), cer_recon.item())
    
    return mel_out, e_loss_g, e_acc_g
      
    
def train_D(args, mel_out, target, target_cl, labels, models, criterions, optimizer_d, trainValid):
    
    (_, model_d, _, _, _) = models
    (_, _, criterion_adv, criterion_cl, _) =  criterions

    if trainValid:
        model_d.train()
    else:
        model_d.eval()
    
    # Adversarial ground truths 1:real, 0: fake
    # valid = torch.ones((len(mel_out), 1), dtype=torch.float32).cuda()
    valid = torch.ones((len(mel_out), 1), dtype=torch.float32).cuda() * 0.9
    fake = torch.zeros((len(mel_out), 1), dtype=torch.float32).cuda()
    
    ###############################
    # Train Discriminator
    ###############################
    
    if trainValid:
        if args.pretrain and args.prefreeze:
            for total_ct, _ in enumerate(model_d.children()):
                ct=0
            for ct, child in enumerate(model_d.children()):
                if ct > total_ct-1: # unfreeze classifier 
                    for param in child.parameters():
                        param.requires_grad = True  # unfreeze D    
        else:
            for p in model_d.parameters():
                p.requires_grad_(True)  # unfreeze D   
                
        # set zero grad
        optimizer_d.zero_grad()

    # run model cl
    real_valid, real_cl = model_d(target)
    fake_valid, fake_cl = model_d(mel_out.detach())

    loss_d_real_valid = criterion_adv(real_valid, valid)
    loss_d_fake_valid = criterion_adv(fake_valid, fake)
    loss_d_real_cl = criterion_cl(real_cl, target_cl)
    
    loss_d_valid = 0.5 * (loss_d_real_valid + loss_d_fake_valid)
    loss_d_cl = loss_d_real_cl
    
    loss_d = args.l_d[0] * loss_d_cl + args.l_d[1] * loss_d_valid
    
    # accuracy
    acc_d_real = (real_valid.round() == valid).float().mean()
    acc_d_fake = (fake_valid.round() == fake).float().mean()
    preds_real = torch.argmax(real_cl,dim=1)
    acc_cl_real = (preds_real == labels).float().mean()
    preds_fake = torch.argmax(fake_cl,dim=1)
    acc_cl_fake = (preds_fake == labels).float().mean()
    
    if trainValid:
        loss_d.backward()
        optimizer_d.step()

    # e_loss_d = (loss_d.item(), loss_d_valid.item(), loss_d_cl.item())
    # 【修改后】增加 real 和 fake 的单独 loss
    e_loss_d = (
        loss_d.item(),  # 0: Total D Loss
        loss_d_valid.item(),  # 1: Total GAN Loss (Real+Fake)/2
        loss_d_cl.item(),  # 2: Classifier Loss
        loss_d_real_valid.item(),  # 3: Real Sample Loss (新增)
        loss_d_fake_valid.item()  # 4: Fake Sample Loss (新增)
    )

    e_acc_d = (acc_d_real.item(), acc_d_fake.item(), acc_cl_real.item(), acc_cl_fake.item())

    return e_loss_d, e_acc_d


# def saveData(args, test_loader, models, epoch, losses):
#
#     model_g = models[0].eval()
#     # model_d = models[1].eval()
#     vocoder = models[2].eval()
#     model_STT = models[3].eval()
#     decoder_STT = models[4]
#
#     input, target, target_cl, voice, data_info = next(iter(test_loader))
#
#     input = input.cuda()
#     target = target.cuda()
#     voice = torch.squeeze(voice, dim=-1).cuda()
#     labels = torch.argmax(target_cl, dim=1)
#
#     with torch.no_grad():
#         # run the mdoel
#         output = model_g(input)
#
#     mel_out = DTW_align(output, target)
#     output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
#
#     # # ✅ 添加这段调试代码
#     # with torch.no_grad():
#     #     min_val = output_denorm.min().item()
#     #     max_val = output_denorm.max().item()
#     #     mean_val = output_denorm.mean().item()
#     #
#     #     print(f"🔍 [DEBUG] Denormed Mel Stats -> Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
#     #
#     #     # HiFi-GAN Universal V1 的典型 Log-Mel 范围参考：
#     #     # 通常在 -5.0 到 0.0 之间 (或者 -10 到 0)
#     #     # 如果看到 Min > 0 或者 Max > 2.0，说明大概率没取 Log，或者是线性值！
#     #     if min_val > 0:
#     #         print("⚠️ 警告：最小值大于 0，这通常不是 Log-Mel 谱的特征！检查是否需要取 Log。")
#     #     if max_val > 5.0:
#     #         print("⚠️ 警告：最大值过大，Vocoder 可能会产生爆音。")
#
#     wav_recon = mel2wav_vocoder(torch.unsqueeze(output_denorm[0],dim=0), vocoder, 1)
#     wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
#
#     wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)
#     if wav_recon.shape[1] !=  voice.shape[1]:
#         p = voice.shape[1] - wav_recon.shape[1]
#         p_s = p//2
#         p_e = p-p_s
#         wav_recon = F.pad(wav_recon, (p_s,p_e))
#
#     ##### STT Wav2Vec 2.0
#     gt_label = args.word_label[labels[0].item()]
#     print(args.word_label)
#     print(gt_label)
#     print('===================================================================')
#
#     transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, gt_label, 1)
#
#     # save
#     wav_recon = np.squeeze(wav_recon.cpu().detach().numpy())
#
#     str_tar = args.word_label[labels[0].item()].replace("|", ",")
#     str_tar = str_tar.replace(" ", ",")
#
#     str_pred = transcript_recon[0].replace("|", ",")
#     str_pred = str_pred.replace(" ", ",")
#
#     title = "Tar_{}-Pred_{}".format(str_tar, str_pred)
#     wavio.write(args.savevoice + '/e{}_{}.wav'.format(str(str(epoch)), title), wav_recon, args.sample_rate_STT, sampwidth=1)

def saveData(args, test_loader, models, epoch, losses):
    model_g = models[0].eval()
    # model_d = models[1].eval()
    vocoder = models[2].eval()
    model_STT = models[3].eval()
    decoder_STT = models[4]

    input, target, target_cl, voice, data_info = next(iter(test_loader))

    input = input.cuda()
    target = target.cuda()
    voice = torch.squeeze(voice, dim=-1).cuda()
    labels = torch.argmax(target_cl, dim=1)

    with torch.no_grad():
        # run the mdoel
        output = model_g(input)

    mel_out = DTW_align(output, target)
    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
    target_denorm = data_denorm(target, data_info[0], data_info[1])

    # --- 生成并保存 Recon (Predicted) Audio ---
    wav_recon = mel2wav_vocoder(torch.unsqueeze(output_denorm[0], dim=0), vocoder, 1)
    wav_recon = torch.reshape(wav_recon, (len(wav_recon), wav_recon.shape[-1]))

    wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)
    if wav_recon.shape[1] != voice.shape[1]:
        p = voice.shape[1] - wav_recon.shape[1]
        p_s = p // 2
        p_e = p - p_s
        wav_recon = F.pad(wav_recon, (p_s, p_e))

    # --- 生成并保存 Target Audio ---
    wav_target = mel2wav_vocoder(torch.unsqueeze(target_denorm[0], dim=0), vocoder, 1)
    wav_target = torch.reshape(wav_target, (len(wav_target), wav_target.shape[-1]))

    wav_target = torchaudio.functional.resample(wav_target, args.sample_rate_mel, args.sample_rate_STT)
    if wav_target.shape[1] != voice.shape[1]:
        p = voice.shape[1] - wav_target.shape[1]
        p_s = p // 2
        p_e = p - p_s
        wav_target = F.pad(wav_target, (p_s, p_e))

    # --- STT Transcription for Recon Audio ---
    ##### STT Wav2Vec 2.0
    gt_label = args.word_label[labels[0].item()]
    print(args.word_label)
    print(gt_label)
    print('===================================================================')

    transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, gt_label, 1)

    # --- Save Files ---
    # Prepare filename components
    str_tar = args.word_label[labels[0].item()].replace("|", ",")
    str_tar = str_tar.replace(" ", ",")

    str_pred = transcript_recon[0].replace("|", ",")
    str_pred = str_pred.replace(" ", ",")

    # Convert to numpy for saving
    wav_recon_np = np.squeeze(wav_recon.cpu().detach().numpy())
    wav_target_np = np.squeeze(wav_target.cpu().detach().numpy())

    # Save reconstructed audio
    title_recon = "e{}_Tar_{}-Pred_{}".format(str(str(epoch)), str_tar, str_pred)
    wavio.write(args.savevoice + '/recon_' + title_recon + '.wav', wav_recon_np, args.sample_rate_STT, sampwidth=1)

    # Save target audio
    title_target = "e{}_Target_{}".format(str(str(epoch)), str_tar)
    wavio.write(args.savevoice + '/target_' + title_target + '.wav', wav_target_np, args.sample_rate_STT, sampwidth=1)


def main(args):
    
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device: {} '.format(torch.cuda.current_device())) # check
    print('The number of available GPU:{}'.format(torch.cuda.device_count()))
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # define generator
    config_file = os.path.join(args.model_config, 'config_g.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_g = AttrDict(json_config)
    model_g = networks.Generator(h_g).cuda()
    
    args.sample_rate_mel = args.sampling_rate
    
    # define discriminator
    config_file = os.path.join(args.model_config, 'config_d.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_d = AttrDict(json_config)
    model_d = networks.Discriminator(h_d).cuda()
    
    # vocoder HiFiGAN
    # LJ_FT_T2_V3/generator_v3,   
    config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    vocoder = model_HiFi(h).cuda()
    state_dict_g = torch.load(args.vocoder_pre)  #, map_location=args.device)
    vocoder.load_state_dict(state_dict_g['generator'])
    
    # STT Wav2Vec
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    model_STT = bundle.get_model().cuda()
    args.sample_rate_STT = bundle.sample_rate
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    args.word_index, args.word_length = word_index(args.word_label, bundle)
    
    # Parallel setting
    model_g = nn.DataParallel(model_g, device_ids=args.gpuNum)
    model_d = nn.DataParallel(model_d, device_ids=args.gpuNum)
    vocoder = nn.DataParallel(vocoder, device_ids=args.gpuNum)
    model_STT = nn.DataParallel(model_STT, device_ids=args.gpuNum)

    # loss function
    criterion_recon = RMSELoss().cuda()
    criterion_adv = nn.BCELoss().cuda()
    criterion_ctc = nn.CTCLoss().cuda()
    criterion_cl = nn.CrossEntropyLoss().cuda()
    CER = CharErrorRate().cuda()

    # optimizer
    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.01)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.lr_d_decay, last_epoch=-1)

   # # create the directory if not exist
   #  if not os.path.exists(args.logDir):
   #      os.mkdir(args.logDir)
   #
   #  # saveDir = args.logDir + args.sub + '_' + args.task
   #  saveDir = args.logDir + '/' + args.sub + '_' + args.task
   #  if not os.path.exists(saveDir):
   #      os.mkdir(saveDir)
   #
   #  args.savevoice = saveDir + '/epovoice'
   #  if not os.path.exists(args.savevoice):
   #      os.mkdir(args.savevoice)
   #
   #  args.savemodel = saveDir + '/savemodel'
   #  if not os.path.exists(args.savemodel):
   #      os.mkdir(args.savemodel)
   #
   #  args.logs = saveDir + '/logs'
   #  if not os.path.exists(args.logs):
   #      os.mkdir(args.logs)

    # create the directory if not exist
    # 1. 获取 logDir 的绝对路径 (防止相对路径带来的 TF 解析问题)
    args.logDir = os.path.abspath(args.logDir)

    if not os.path.exists(args.logDir):
        os.makedirs(args.logDir, exist_ok=True)

    # 2. 构建 saveDir (此时 logDir 已经是绝对路径)
    saveDir = os.path.join(args.logDir, f"{args.sub}_{args.task}")

    if not os.path.exists(saveDir):
        os.makedirs(saveDir, exist_ok=True)

    # 3. 定义子目录
    args.savevoice = os.path.join(saveDir, 'epovoice')
    args.savemodel = os.path.join(saveDir, 'savemodel')
    args.logs = os.path.join(saveDir, 'logs')

    # 4. 统一创建子目录
    for dir_path in [args.savevoice, args.savemodel, args.logs]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    # 【关键修复】确保传给 SummaryWriter 的绝对是绝对路径
    args.logs = os.path.abspath(args.logs)

    print(f"[Debug] Final Absolute Log Path: {args.logs}")
    print(f"[Debug] Directory Exists? {os.path.exists(args.logs)}")

        
    # Load trained model
    start_epoch = 0
    if args.pretrain:
        loc_g = os.path.join(args.trained_model, 'checkpoint_g.pt')
        loc_d = os.path.join(args.trained_model, 'checkpoint_d.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            print("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_d))

    if args.resume:
        loc_g = os.path.join(args.savemodel, 'checkpoint_g.pt')
        loc_d = os.path.join(args.savemodel, 'checkpoint_d.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
            start_epoch = checkpoint_g['epoch'] + 1
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            print("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_d))

    # Tensorboard setting
    args.writer = SummaryWriter(args.logs)
    
    # Data loader define
    generator = torch.Generator().manual_seed(args.seed)

    trainset = myDataset(mode=0, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.recon)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)
    
    valset = myDataset(mode=2, data=args.dataLoc+'/'+args.sub, task=args.task, recon=args.recon)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)

    epoch = start_epoch
    lr_g = 0
    lr_d = 0
    best_loss = 1000
    is_best = False
    epochs_since_improvement = 0
    
    for epoch in range(start_epoch, args.max_epochs):
        
        start_time = time.time()

        # 【修改】先打印当前 LR，确认是否正确加载
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        print(f"✅ [START EPOCH {epoch}] Current LR G: {current_lr_g:.8f}, D: {current_lr_d:.8f}")
        
        for param_group in optimizer_g.param_groups:
            lr_g = param_group['lr']
        for param_group in optimizer_d.param_groups:
            lr_d = param_group['lr']

        # scheduler_g.step(epoch)
        # scheduler_d.step(epoch)

        print("Epoch : %d/%d" %(epoch, args.max_epochs) )
        print("Learning rate for G: %.9f" %lr_g)
        print("Learning rate for D: %.9f" %lr_d)

        Tr_losses = train(args, train_loader, 
                          (model_g, model_d, vocoder, model_STT, decoder_STT), 
                          (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER), 
                          (optimizer_g, optimizer_d), 
                          epoch,
                          True)

        # 2. 【核心修改】条件执行验证 (Validate)
        # 条件：(epoch + 1) 能被 interval 整除 OR 是最后一个 epoch
        is_val_epoch = ((epoch + 1) % args.val_interval == 0) or (epoch == args.max_epochs - 1)

        if is_val_epoch:
            print(f"\n🔍 [Epoch {epoch}] Running Validation...")
            Val_losses = train(args, val_loader,
                               (model_g, model_d, vocoder, model_STT, decoder_STT),
                               (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER),
                               ([], []),  # 验证时不传优化器
                               epoch,
                               False)

            # 更新学习率 (通常在验证后或 epoch 结束时更新)
            scheduler_g.step()
            scheduler_d.step()
            next_lr_g = optimizer_g.param_groups[0]['lr']
            print(f"✅ [END EPOCH {epoch}] Next LR G: {next_lr_g:.8f}")

            # --- 评估与保存逻辑 (仅在验证 epoch 执行) ---

            # Did validation loss improve?
            # 注意：Val_losses[0] 是总 loss
            loss_total = Val_losses[0]
            is_best = loss_total < best_loss
            best_loss = min(loss_total, best_loss)

            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0
                print("✨ New Best Model Saved!")

            # 保存模型 checkpoint
            state_g = {'arch': str(model_g),
                       'state_dict': model_g.state_dict(),
                       'epoch': epoch,
                       'optimizer_state_dict': optimizer_g.state_dict()}

            state_d = {'arch': str(model_d),
                       'state_dict': model_d.state_dict(),
                       'epoch': epoch,
                       'optimizer_state_dict': optimizer_d.state_dict()}

            save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
            save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')

            # 保存音频样本 (仅在验证时做，因为很耗时)
            saveData(args, val_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), epoch,
                     (Tr_losses, Val_losses))

        else:
            # 如果不验证，只更新学习率 (或者你也可以选择在不验证的 epoch 不更新 LR，视策略而定)
            # 通常 ExponentialLR 是每个 epoch 都步进的，无论是否验证
            scheduler_g.step()
            scheduler_d.step()
            print(f"⏩ [Epoch {epoch}] Skipped Validation. Next LR G: {optimizer_g.param_groups[0]['lr']:.8f}")

            # # 即使不验证，也保存最新的 checkpoint (方便中断恢复)，但不标记为 best
            # state_g = {'arch': str(model_g),
            #            'state_dict': model_g.state_dict(),
            #            'epoch': epoch,
            #            'optimizer_state_dict': optimizer_g.state_dict()}
            # state_d = {'arch': str(model_d),
            #            'state_dict': model_d.state_dict(),
            #            'epoch': epoch,
            #            'optimizer_state_dict': optimizer_d.state_dict()}
            #
            # # 这里保存为普通 checkpoint，不触发 is_best 逻辑覆盖 best_model
            # save_checkpoint(state_g, False, args.savemodel, 'checkpoint_g.pt')
            # save_checkpoint(state_d, False, args.savemodel, 'checkpoint_d.pt')

        time_taken = time.time() - start_time
        print("Time: %.2f\n" % time_taken)

        # Val_losses = train(args, val_loader,
        #                    (model_g, model_d, vocoder, model_STT, decoder_STT),
        #                    (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER),
        #                    ([],[]),
        #                    epoch,
        #                    False)
        #
        # # # 【修改】在 epoch 结束时更新学习率
        # scheduler_g.step()
        # scheduler_d.step()
        #
        # # 打印更新后的 LR 供检查
        # next_lr_g = optimizer_g.param_groups[0]['lr']
        # print(f"✅ [END EPOCH {epoch}] Next LR G: {next_lr_g:.8f}")
        #
        # # Save checkpoint
        # state_g = {'arch': str(model_g),
        #          'state_dict': model_g.state_dict(),
        #          'epoch': epoch,
        #          'optimizer_state_dict': optimizer_g.state_dict()}
        #
        # state_d = {'arch': str(model_d),
        #          'state_dict': model_d.state_dict(),
        #          'epoch': epoch,
        #          'optimizer_state_dict': optimizer_d.state_dict()}
        #
        # # Did validation loss improve?
        # loss_total = Val_losses[0]
        # is_best = loss_total < best_loss
        # best_loss = min(loss_total, best_loss)
        #
        # if not is_best:
        #     epochs_since_improvement += 1
        #     print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        # else:
        #     epochs_since_improvement = 0
        #
        # save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
        # save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')
        #
        # saveData(args, val_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), epoch, (Tr_losses,Val_losses))
        #
        # time_taken = time.time() - start_time
        # print("Time: %.2f\n"%time_taken)
        
    args.writer.flush()

if __name__ == '__main__':

    # dataDir = './sample_data'
    dataDir = './processed_dataset_1_orign'
    logDir = './TrainResult_orign'
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--vocoder_pre', type=str, default='./pretrained_model/UNIVERSAL_V1/g_02500000', help='pretrained vocoder file path')
    parser.add_argument('--trained_model', type=str, default='./pretrained_model', help='trained model for G & D folder path')
    parser.add_argument('--model_config', type=str, default='./models', help='config for G & D folder path')
    parser.add_argument('--dataLoc', type=str, default=dataDir)
    parser.add_argument('--config', type=str, default='./config_myPrivate1.json')
    parser.add_argument('--logDir', type=str, default=logDir)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--prefreeze', type=bool, default=False)
    # parser.add_argument('--gpuNum', type=list, default=[0, 1, 2])
    parser.add_argument('--gpuNum', type=list, default=[0])
    # parser.add_argument('--batch_size', type=int, default=52)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--sub', type=str, default='sub1')
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--recon', type=str, default='Voice_mel')
    parser.add_argument('--unseen', type=str, default='none')


    # 【新增】验证间隔参数：每隔多少个 epoch 验证一次 (默认 5)
    parser.add_argument('--val_interval', type=int, default=5, help='Validate every N epochs')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    main(args)        
    
    
    