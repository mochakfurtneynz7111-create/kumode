import os
from os.path import join as j_
import pdb
import shap
import torch.nn.functional as F

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


import numpy as np
import torch
import torch.nn as nn
import shap
import pickle
try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print('scikit-survival not installed. Exiting...')
    raise

from mil_models.tokenizer import PrototypeTokenizer
from mil_models import create_multimodal_survival_model, prepare_emb
from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from utils.utils import (EarlyStopping, save_checkpoint, AverageMeter, safe_list_to,
                         get_optim, print_network, get_lr_scheduler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROTO_MODELS = ['PANTHER', 'OT', 'H2T', 'ProtoCount']

def train(datasets, args):
    """
    Train for a single fold for suvival
    """
    
    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    assert args.es_metric in ['loss', 'c_index']
    
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()

    args.feat_dim = args.in_dim # Patch feature dimension
    print('\nInit Model...', end=' ')

    # If prototype-based models, need to create slide-level embeddings
    if args.model_histo_type in PROTO_MODELS:
        datasets, _ = prepare_emb(datasets, args, mode='survival')

        new_in_dim = None
        for k, loader in datasets.items():
            assert loader.dataset.X is not None
            new_in_dim_curr = loader.dataset.X.shape[-1]
            if new_in_dim is None:
                new_in_dim = new_in_dim_curr
            else:
                assert new_in_dim == new_in_dim_curr

            # The original embedding is 1-D (long) feature vector
            # Reshape it to (n_proto, -1)
            tokenizer = PrototypeTokenizer(args.model_histo_type, args.out_type, args.n_proto)
            prob, mean, cov = tokenizer(loader.dataset.X)
            loader.dataset.X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)

            factor = args.n_proto
            
        args.in_dim = new_in_dim // factor
    else:
        print(f"{args.model_histo_type} doesn't construct unsupervised slide-level embeddings!")

    ## Set the dimensionality for different inputs
    args.omic_dim = datasets['train'].dataset.omics_data.shape[1]

    if args.omics_modality in ['pathway', 'functional']:
        omic_sizes = datasets['train'].dataset.omic_sizes
    else:
        omic_sizes = []
        
    # 创建模型时
    transformer_config = {
        'use_graph_attn': args.use_graph_attn,
        'use_spatial_ppeg': args.use_spatial_ppeg,
        'use_spatial_attn': args.use_spatial_attn
    }

    model = create_multimodal_survival_model(args, omic_sizes=omic_sizes,transformer_config=transformer_config)

    model.to(device)
    
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    if args.early_stopping:
        print('\nSetup EarlyStopping...', end=' ')
        early_stopper = EarlyStopping(save_dir=args.results_dir,
                                      patience=args.es_patience,
                                      min_stop_epoch=args.es_min_epochs,
                                      better='min' if args.es_metric == 'loss' else 'max',
                                      verbose=True)
    else:
        print('\nNo EarlyStopping...', end=' ')
        early_stopper = None

    # 在训练循环的适当位置，例如每个折的结束时
    with open('processed_train_dataset.pkl', 'wb') as f:
        pickle.dump(datasets['train'], f)

    with open('processed_test_dataset.pkl', 'wb') as f:
        pickle.dump(datasets['test'], f)
    print("成功保存数据")
    
    #####################
    # The training loop #
    #####################
    for epoch in range(args.max_epochs):
        step_log = {'epoch': epoch, 'samples_seen': (epoch + 1) * len(datasets['train'].dataset)}

        ### Train Loop
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        train_results = train_loop_survival(model, datasets['train'], optimizer, lr_scheduler, loss_fn,
                                            print_every=args.print_every, accum_steps=args.accum_steps)


        ### Validation Loop (Optional)
        if 'val' in datasets.keys():
            print('#' * 11, f'VAL Epoch: {epoch}', '#' * 11)
            val_results, _ = validate_survival(model, datasets['val'], loss_fn,
                                                   print_every=args.print_every, verbose=True)

            ### Check Early Stopping (Optional)
            if early_stopper is not None:
                # ✅ 支持 c_index
                if args.es_metric == 'loss':
                    score = val_results['loss']
                elif args.es_metric == 'c_index':
                    score = val_results['c_index']  # 🔥 添加这个分支
                else:
                    raise NotImplementedError
                    
                save_ckpt_kwargs = dict(config=vars(args),
                                        epoch=epoch,
                                        model=model,
                                        score=score,
                                        fname=f's_checkpoint.pth')
                stop = early_stopper(epoch, score, save_checkpoint, save_ckpt_kwargs)
                if stop:
                    break
        print('#' * (22 + len(f'TRAIN Epoch: {epoch}')), '\n')

    ### End of epoch: Load in the best model (or save the latest model with not early stopping)
    if args.early_stopping:
        model.load_state_dict(torch.load(j_(args.results_dir, f"s_checkpoint.pth"))['model'])
    else:
        torch.save(model.state_dict(), j_(args.results_dir, f"s_checkpoint.pth"))


    ### End of epoch: Evaluate on val and test set
    results, dumps = {}, {}
    for k, loader in datasets.items():
        # 以下修改成只在验证集上测试
        print(f'End of training. Evaluating on Split {k.upper()}...:')
        return_attn = True # True for MMP
        results[k], dumps[k] = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                    dump_results=True, return_attn=return_attn, verbose=False)

        if k == 'train':
            _ = results.pop('train')  # Train results by default are not saved in the summary, but train dumps are
        
    # writer.close()
    return results, dumps

## SURVIVAL
def train_loop_survival(model, loader, optimizer, lr_scheduler, loss_fn=None, 
                        print_every=50, accum_steps=32):
    
    model.train()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    
    for batch_idx, batch in enumerate(loader):
        data = safe_list_to(batch['img'], device)
        label = safe_list_to(batch['label'], device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None

        omics = safe_list_to(batch['omics'], device)

        out, log_dict = model(data, omics, attn_mask=attn_mask, label=label, censorship=censorship, loss_fn=loss_fn)
 
        if out['loss'] is None:
            continue

        # Get loss + backprop
        loss = out['loss']
        loss = loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # End of iteration survival-specific metrics to calculate / log
        all_risk_scores.append(out['risk'].detach().cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))

        bag_size_meter.update(data.size(1), n=len(data))

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    return results


@torch.no_grad()
def validate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      return_attn=False,
                      verbose=1):
    model.eval()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_omic_attn, all_cross_attn, all_path_attn = [], [], []

    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:  # 只打印第一个 batch 的 sample IDs
            sample_ids = [loader.dataset.get_sample_id(idx) for idx in range(len(batch['img']))]
            print("Sample IDs in the first test batch:", sample_ids)
        data = batch['img'].to(device)
        label = batch['label'].to(device)
        omics = safe_list_to(batch['omics'], device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        
        out, log_dict = model(data, omics, attn_mask=attn_mask, label=label, censorship=censorship, loss_fn=loss_fn, return_attn=return_attn)
        if return_attn:
            omic_attn = out['omic_attn'].detach().cpu().numpy()
            cross_attn = out['cross_attn'].detach().cpu().numpy()
            path_attn = out['path_attn'].detach().cpu().numpy()

            # 确保都是3维 (batch_size, n_paths, dim)
            if omic_attn.ndim == 2:
                omic_attn = omic_attn[np.newaxis, ...]
                cross_attn = cross_attn[np.newaxis, ...]
                path_attn = path_attn[np.newaxis, ...]

            all_omic_attn.append(omic_attn)
            all_cross_attn.append(cross_attn)
            all_path_attn.append(path_attn)
            print(f"Batch {batch_idx}: omic_attn shape = {out['omic_attn'].shape}, batch_size = {len(data)}")
            """
            all_omic_attn.append(out['omic_attn'].detach().cpu().numpy())
            all_cross_attn.append(out['cross_attn'].detach().cpu().numpy())
            all_path_attn.append(out['path_attn'].detach().cpu().numpy())
            """
        # End of iteration survival-specific metrics to calculate / log
        bag_size_meter.update(data.size(1), n=len(data))
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        all_risk_scores.append(out['risk'].cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    if return_attn:
        if len(all_omic_attn[0].shape) == 2:
            all_omic_attn = np.stack(all_omic_attn)
            all_cross_attn = np.stack(all_cross_attn)
            all_path_attn = np.stack(all_path_attn)
        else:
            all_omic_attn = np.vstack(all_omic_attn)
            all_cross_attn = np.vstack(all_cross_attn)
            all_path_attn = np.vstack(all_path_attn)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print(c_index)
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    print(results)

    """
    if recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        surv_loss_dict = loss_fn(logits=torch.tensor(all_risk_scores).unsqueeze(1),
                                 times=torch.tensor(all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})
    """
    
    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    sample_ids = np.array(loader.dataset.idx2sample_df['sample_id'])
    
    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = sample_ids 
        # dumps['sample_ids'] = np.array(loader.dataset.idx2sample_df['sample_id'])
        if return_attn:
            dumps['all_omic_attn'] = all_omic_attn
            dumps['all_cross_attn'] = all_cross_attn
            dumps['all_path_attn'] = all_path_attn

     # 添加 Kaplan-Meier 分析和绘图\

    km_censorships = 1 - all_censorships 
    # plot_km_curve(all_risk_scores, km_censorships, all_event_times, dumps['sample_ids'])
    plot_km_curve(all_risk_scores, km_censorships, all_event_times, sample_ids)

    return results, dumps

def plot_km_curve(risk_scores, censorships, event_times, sample_ids):
    """
    绘制 Kaplan-Meier 生存曲线
    """
    """
    print(risk_scores.shape)
    print("risk_scores.shape")    
    print(risk_scores)
    print("risk_scores")
    """
    median_risk = np.median(risk_scores)
    """
    print(median_risk)
    print("median_risk")
    """
    high_risk = risk_scores > median_risk
    low_risk = risk_scores <= median_risk

    #  打印高风险和低风险组的事件时间和截尾情况
    """
    print("High risk event times:", event_times[high_risk])
    print("High risk censorships:", censorships[high_risk])
    print("Low risk event times:", event_times[low_risk])
    print("Low risk censorships:", censorships[low_risk])
    # 获取高风险和低风险组的样本 ID
    high_risk_ids = sample_ids[high_risk]
    low_risk_ids = sample_ids[low_risk]
     # 打印高风险和低风险组的样本 ID
    print("High risk sample IDs:", high_risk_ids)
    print("Low risk sample IDs:", low_risk_ids)
    """


    
     # 进行 Logrank 检验并获取 p 值
    results_logrank = logrank_test(event_times[high_risk], event_times[low_risk],
                                    event_observed_A=censorships[high_risk],
                                    event_observed_B=censorships[low_risk])
   
    # 创建 Kaplan-Meier 估计器
    kmf = KaplanMeierFitter()

    # groups = np.unique(censorships)
    # for group in groups:
    #     # print(f"Group {group} has {np.sum(censorships == group)} samples.")
    #     is_group = censorships == group
    #     # print(event_times[is_group])
    #     # print("event_times[is_group]")
    #     kmf.fit(event_times[is_group], censorships[is_group], label=f"Status {group}")
    #     # 计算生存曲线的时间和生存概率
    #     survival_prob = kmf.survival_function_.values
    #     time = kmf.survival_function_.index
    
    #     # 绘制生存曲线
    #     plt.step(time, survival_prob, where="post", label=f"Status {group}")
    

    

    # 打印 p 值
    print("Logrank test p-value:", results_logrank.p_value)

    kmf.fit(event_times[high_risk], censorships[high_risk], label="High Risk Group")
    plt.step(kmf.survival_function_.index, kmf.survival_function_.values, where="post", label="High Risk Group")
    
    kmf.fit(event_times[low_risk], censorships[low_risk], label="Low Risk Group")
    plt.step(kmf.survival_function_.index, kmf.survival_function_.values, where="post", label="Low Risk Group")
    
    
    

    # 在图中添加 p 值文本
    p_value_text = f"p = {results_logrank.p_value:.4e}"
    plt.text(0.05, 0.1, p_value_text, transform=plt.gca().transAxes, fontsize=14)

    # 设置图例、标签和标题
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier Survival Curve")
    # plt.grid(True)

    # 定义存储路径和文件名
    file_name = "KM_Survival_Curve.png"

    # 保存图像
    plt.savefig(file_name, dpi=400)
    print(f"Kaplan-Meier plot saved to {file_name}")

    # 显示图形
    plt.show()

