

CUDA_VISIBLE_DEVICES=2 python -m opensora.eval.fid.step2_run_model \
--real_images "/storage/dataset/val2017" \
--fake_images "/storage/ongoing/12.29/eval/t2i_ablation_arch_gen_s24_nocfg/umt5/base/COCO2017" \
--batch_size 64 

CUDA_VISIBLE_DEVICES=2 python -m opensora.eval.fid.step2_run_model \
--real_images "/storage/dataset/ImageNet-2012/ILSVRC/Data/CLS-LOC/val" \
--fake_images "/storage/ongoing/12.29/eval/t2i_ablation_arch_gen_s24_nocfg/umt5/base/ImageNet" \
--batch_size 64

