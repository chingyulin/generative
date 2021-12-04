# python srresnet_trainer.py --save_dir doc_mse  --dataset document
# python srresnet_trainer.py --save_dir doc_mse_vgg  --dataset document --vgg_loss
# python srresnet_trainer.py --save_dir doc_mse_hed  --dataset document --hed_loss
# python srresnet_trainer.py --save_dir doc_mse_gan  --dataset document --adv_loss --adv_type nonsaturating
# python srresnet_trainer.py --save_dir doc_mse_wgan  --dataset document --adv_loss --adv_type wasserstein
python srresnet_trainer.py --save_dir doc_mse_hed_vgg_long  --dataset document --hed_loss --vgg_loss
# python srresnet_trainer.py --save_dir doc_mse_hed_wgan  --dataset document --hed_loss --adv_loss --adv_type wasserstein
# python srresnet_trainer.py --save_dir doc_mse_hed_gan  --dataset document --hed_loss --adv_loss --adv_type nonsaturating
# python srresnet_trainer.py --save_dir doc_mse_hed_vgg_wgan  --dataset document --hed_loss --vgg_loss --adv_loss --adv_type wasserstein
# python srresnet_trainer.py --save_dir doc_mse_hed_vgg_gan  --dataset document --hed_loss --vgg_loss --adv_loss --adv_type nonsaturating


