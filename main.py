import hydra
from omegaconf import DictConfig, OmegaConf
from data.dataset import *
from network.s3ngan import *


@hydra.main(config_path='./configs', config_name='config.yaml')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    train_dataset = alignedstainData(cfg.dataset.opt_dataset)
    test_dataset = singlestainData(cfg.dataset.opt_dataset, stage='test')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.run.opt_run['batchSize'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.run.opt_run['batchSize'],
        shuffle=False,
        num_workers=8,
        drop_last=True)
    print('total train: ', len(train_loader))
    print('total test: ', len(test_loader))
    model = twodecoderGANModel(cfg)
    overall_best_psnr = 0
    overall_best_ssim = 0
    overall_best_epoch = 0
    start_epoch = cfg.run.opt_run['which_epoch'] + 1 if cfg.run.opt_run['continue_train'] else cfg.run.opt_run['which_epoch']
    if cfg.run.opt_run['stage'] == 'test':
        print('Start testing...')
        model.test(test_loader, stage='test', save=True)
        NMI('/home/congz3414050/cong/media/results/%s/normalised_cam16/%d/test'%(cfg.dataset.opt_dataset['name'], cfg.run.opt_run['which_epoch']))
    else:
        print('Start training...')
        for epoch in range(start_epoch, cfg.run.opt_run['n_epoch'] + 1):
            epoch_loss = {}
            iters = 0
            for i, (inputs) in tqdm(enumerate(train_loader)):
                model.set_input(inputs)
                model.optimize_parameters()
                current_error = model.get_current_errors()
                for loss_name in current_error:
                    if loss_name not in epoch_loss:
                        epoch_loss[loss_name] = current_error[loss_name]
                    else:
                        epoch_loss[loss_name] += current_error[loss_name]
                iters += 1
                break
            output = "===> Epoch {%d} Complete: Avg." % epoch
            for loss_name in epoch_loss:
                output += '%s: %.3f ' % (loss_name, epoch_loss[loss_name] / iters)
            print(output)
            adjust_learning_rate(model.optimizer_D, epoch, cfg.run.opt_run, cfg.run.opt_run['lr_D'])
            adjust_learning_rate(model.optimizer_G, epoch, cfg.run.opt_run, cfg.run.opt_run['lr_G'])

            model.get_current_visuals(epoch)
            model.save(epoch)
            epoch_psnr, epoch_ssim = model.test(test_loader)
            if epoch_psnr > overall_best_psnr and epoch_ssim > overall_best_ssim:
                overall_best_psnr = epoch_psnr
                overall_best_ssim = epoch_ssim
                overall_best_epoch = epoch
            print('Current Best: Epoch@%d, PSNR:%.3f; SSIM:%.3f' % (overall_best_epoch,
                                                                    overall_best_psnr,
                                                                    overall_best_ssim))

if __name__ == '__main__':
    main()