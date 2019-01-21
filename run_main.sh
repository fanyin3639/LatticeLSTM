export CUDA_VISIBLE_DEVICES=1
python main.py --status train \
		--train data/ResumeNER/train.char.bmes \
		--dev data/ResumeNER/dev.char.bmes \
		--test data/ResumeNER/test.char.bmes \
		--savemodel data/ResumeNER/saved_model \


# python main.py --status decode \
# 		--raw data/ResumeNER/test.char.bmes \
# 		--savedset data/ResumeNER/saved_model \
# 		--loadmodel data/ResumeNER/saved_model.13.model \
# 		--output data/ResumeNER/raw.out \
