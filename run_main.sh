# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1
python main.py


# python main.py --status decode \
# 		--raw data/ResumeNER/test.char.bmes \
# 		--savedset data/ResumeNER/saved_model \
# 		--loadmodel data/ResumeNER/saved_model.13.model \
# 		--output data/ResumeNER/raw.out \
