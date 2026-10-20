import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import util.util as util
from evaluations.fid_kid import (
    InceptionV3,
    calculate_fid,
    calculate_kid,
    collect_images,
    get_activations,
)


@torch.no_grad()
def _extract_fake_val_features(model, val_dataset, val_dataset2, metric_model, metric_device,
                               dims=2048, image_size=299, metric_batch_size=32):
    target_count = getattr(val_dataset.dataset, 'A_size', len(val_dataset))
    features = []
    pending = []
    pending_count = 0
    seen = 0

    def flush_pending():
        nonlocal pending, pending_count
        if pending_count == 0:
            return
        batch = torch.cat(pending, dim=0).to(metric_device, dtype=torch.float32, non_blocking=True)
        pred = metric_model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        features.append(pred.detach().cpu().numpy().reshape(pred.size(0), -1))
        pending = []
        pending_count = 0

    for data, data2 in zip(val_dataset, val_dataset2):
        remaining = target_count - seen
        if remaining <= 0:
            break

        model.set_input(data, data2)
        model.inference()

        fake = model.fake_B.detach()
        if fake.size(0) > remaining:
            fake = fake[:remaining]

        fake = (fake.clamp(-1, 1) + 1.0) / 2.0
        if image_size > 0 and (fake.shape[-2] != image_size or fake.shape[-1] != image_size):
            fake = F.interpolate(fake, size=(image_size, image_size), mode='bilinear', align_corners=False)
        fake = fake.to(metric_device, dtype=torch.float32, non_blocking=True)
        pending.append(fake)
        pending_count += fake.size(0)
        seen += fake.size(0)
        if pending_count >= metric_batch_size:
            flush_pending()

    flush_pending()

    if not features:
        raise RuntimeError('No validation fake_B features were extracted.')
    return np.concatenate(features, axis=0)[:, :dims]


def initialize_fid_kid_metrics(opt, val_dataset, metric_device):
    dims = 2048
    image_size = 299
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    metric_model = InceptionV3([block_idx], resize_input=False, normalize_input=True).to(metric_device)
    metric_model.eval()

    real_dir = getattr(val_dataset.dataset, 'dir_B', os.path.join(opt.dataroot, 'valB'))
    real_files = collect_images(real_dir, '*', recursive=False)
    if len(real_files) == 0:
        raise RuntimeError('No real valB images found for FID/KID: %s' % real_dir)

    print('Caching real valB Inception features for FID/KID: %d images from %s' % (len(real_files), real_dir))
    real_features = get_activations(
        real_files, metric_model, batch_size=32, dims=dims,
        device=metric_device, image_size=image_size
    )

    metric_dir = os.path.join(opt.checkpoints_dir, opt.name, 'val_metrics')
    os.makedirs(metric_dir, exist_ok=True)
    log_path = os.path.join(metric_dir, 'fid_kid_log.txt')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('iters\tfid\tkid\tkid_std\tkid_x1000\tkid_std_x1000\tkid_subset_size\tn_real\tn_fake\n')

    return {
        'model': metric_model,
        'device': metric_device,
        'real_features': real_features,
        'real_count': len(real_files),
        'dims': dims,
        'image_size': image_size,
        'metric_batch_size': 32,
        'kid_subsets': 100,
        'kid_subset_size': 100,
        'log_path': log_path,
    }


def compute_full_val_fid_kid(model, val_dataset, val_dataset2, metric_state, total_iters):
    fake_features = _extract_fake_val_features(
        model, val_dataset, val_dataset2,
        metric_state['model'], metric_state['device'],
        dims=metric_state['dims'], image_size=metric_state['image_size'],
        metric_batch_size=metric_state['metric_batch_size']
    )

    fid = calculate_fid(metric_state['real_features'], fake_features)
    kid_mean, kid_std, kid_subset_size = calculate_kid(
        metric_state['real_features'], fake_features,
        subset_size=metric_state['kid_subset_size'],
        n_subsets=metric_state['kid_subsets'],
        seed=1234 + int(total_iters),
    )

    with open(metric_state['log_path'], 'a') as f:
        f.write('%d\t%.8f\t%.10f\t%.10f\t%.6f\t%.6f\t%d\t%d\t%d\n' % (
            total_iters, fid, kid_mean, kid_std,
            kid_mean * 1000.0, kid_std * 1000.0,
            kid_subset_size, metric_state['real_count'], fake_features.shape[0]
        ))

    print('VAL FID/KID @ %d: FID %.6f | KID %.8f +/- %.8f | KIDx1000 %.6f +/- %.6f | n_real %d n_fake %d' % (
        total_iters, fid, kid_mean, kid_std, kid_mean * 1000.0, kid_std * 1000.0,
        metric_state['real_count'], fake_features.shape[0]
    ))
    print('FID/KID log saved to: %s' % metric_state['log_path'])
    return fid


def run_validation(model, val_dataset, val_dataset2, save_dir, total_iters, metric_state=None):
    iter_dir = os.path.join(save_dir, 'iter_%d' % total_iters)
    os.makedirs(iter_dir, exist_ok=True)
    model.eval()
    fid = None
    if hasattr(model, 'ema_swap_in'):
        model.ema_swap_in()
    try:
        with torch.no_grad():
            for i, (data, data2) in enumerate(zip(val_dataset, val_dataset2)):
                if i >= 16:
                    break
                model.set_input(data, data2)
                model.inference()


                for label in ('real_A', 'fake_B', 'real_B'):
                    if hasattr(model, label):
                        im = util.tensor2im(getattr(model, label))
                        path = os.path.join(iter_dir, '%02d_%s.png' % (i, label))
                        util.save_image(im, path)
            if metric_state is not None:
                fid = compute_full_val_fid_kid(model, val_dataset, val_dataset2, metric_state, total_iters)
    finally:
        if hasattr(model, 'ema_swap_out'):
            model.ema_swap_out()
    for name in model.model_names:
        if isinstance(name, str):
            getattr(model, 'net' + name).train()
    return fid


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    dataset_size = len(dataset)


    val_opt = util.copyconf(opt, phase="val", serial_batches=True,
                            no_flip=True, load_size=opt.crop_size, isTrain=False)
    val_dataset = create_dataset(val_opt)
    val_dataset2 = create_dataset(val_opt)
    val_save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'val_images')
    os.makedirs(val_save_dir, exist_ok=True)
    val_freq = 2000

    model = create_model(opt)
    print('The number of training images = %d' % dataset_size)

    metric_device = torch.device('cuda:%d' % opt.gpu_ids[0] if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')
    metric_state = initialize_fid_kid_metrics(opt, val_dataset, metric_device)

    visualizer = Visualizer(opt)
    opt.visualizer = visualizer
    total_iters = 0
    best_fid = float('inf')

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        dataset.set_epoch(epoch)
        for i, (data,data2) in enumerate(zip(dataset,dataset2)):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data,data2)
                model.setup(opt)
                model.parallelize()
            model.set_input(data,data2)
            model.optimize_parameters()
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % val_freq == 0:
                print('Running validation (total_iters %d)' % total_iters)
                val_fid = run_validation(model, val_dataset, val_dataset2, val_save_dir, total_iters, metric_state)
                if val_fid is not None and val_fid < best_fid:
                    best_fid = float(val_fid)
                    print('New best FID %.6f @ %d -> saving best checkpoint' % (best_fid, total_iters))
                    if hasattr(model, 'ema_swap_in'):
                        model.ema_swap_in()
                    model.save_networks('best')
                    if hasattr(model, 'ema_swap_out'):
                        model.ema_swap_out()
                    with open(os.path.join(opt.checkpoints_dir, opt.name, 'best_fid.txt'), 'w') as bf:
                        bf.write('%d\t%.6f\n' % (total_iters, best_fid))

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
