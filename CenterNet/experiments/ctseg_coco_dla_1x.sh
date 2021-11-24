cd src
# train
python main.py ctseg --exp_id coco_ctseg_dla_1x --batch_size 128 --master_batch 10 --lr 1e-4 --load_model ../models/ctdet_coco_dla_1x.pth --gpus 0,1,2,3,4,5,6,7 --num_workers 16  --num_epochs 12 --lr_step 10
# test
python test.py ctseg --exp_id coco_ctseg_dla_1x --keep_res --resume
cd ..

# vis
# python demo.py ctseg --exp_id coco_ctseg_dla_1x --keep_res --resume --demo ../data/coco/val2017 --debug 4
