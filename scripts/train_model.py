
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from muss.fairseq.main import fairseq_train_and_evaluate_with_parametrization
from muss.mining.training import get_bart_kwargs, get_score_rows, get_mbart_trimmed_kwargs
from muss.resources.prepare import prepare_wikilarge_detokenized, prepare_asset

from muss.muss.mining.training import get_mbart_kwargs
from muss.resources.datasets import create_smaller_dataset

train_lang = 'ger_mbart_head_middle'

if train_lang == 'wiki':
    # This dataset should exist in resources/datasets/ and contain the following files:
    # train.complex, train.simple, valid.complex, valid.simple, test.complex, test.simple
    prepare_wikilarge_detokenized()
    prepare_asset()
    dataset = 'wikilarge_detokenized-lines100'
    kwargs = get_bart_kwargs(dataset=dataset, language='en', use_access=True, bart_model='bart.large')
    kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
    kwargs['train_kwargs']['max_tokens'] = 928  # Lower this number to prevent OOM
    kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/resources/models/bart.large/model_928.pt'
    # kwargs['train_kwargs']['memory_efficient_fp16'] = True  # Lower this number to prevent OOM
    kwargs['train_kwargs']['max_source_positions'] = 928  # Lower this number to prevent OOM
    kwargs['train_kwargs']['max_target_positions'] = 928  # Lower this number to prevent OOM
    kwargs['train_kwargs']['update_freq'] = 16  # Lower this number to prevent OOM
    kwargs['train_kwargs']['batch_size'] = 16  # Lower this number to prevent OOM
    result = fairseq_train_and_evaluate_with_parametrization(**kwargs)

    #create_smaller_dataset(dataset, 100)


elif train_lang == 'ger_base':
    #### ger
    dataset = 'uts_de_query-e978f9aa59801cf53e6c41b6366089fc_db-e978f9aa59801cf53e6c41b6366089fc_topk-8_nprobe-16_density-0.6_distance-0.05_filter_ne-False_levenshtein-0.2_simplicity-0.0'
    kwargs = get_bart_kwargs(dataset=dataset, language='en', use_access=True, bart_model='bart.large')
    kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
    # kwargs['train_kwargs']['update_freq'] = 16  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['batch_size'] = 16  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['batch_size'] = 64  # Lower this number to prevent OOM
    kwargs['train_kwargs']['memory_efficient_fp16'] = True
    kwargs['train_kwargs']['max_sentences'] = 32  # Lower this number to prevent OOM
    kwargs['train_kwargs']['max_tokens'] = 1024  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['max_tokens'] = 3000  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1650634359410/checkpoints/checkpoint_31_1000.pt'

    # max tokens per GPU = 4096 and max sentences per GPU = 64
    result = fairseq_train_and_evaluate_with_parametrization(**kwargs)

elif train_lang == 'ger_base_extend':
    #### ger: continue training at last update
    dataset = 'uts_de_query-e978f9aa59801cf53e6c41b6366089fc_db-e978f9aa59801cf53e6c41b6366089fc_topk-8_nprobe-16_density-0.6_distance-0.05_filter_ne-False_levenshtein-0.2_simplicity-0.0'
    kwargs = get_bart_kwargs(dataset=dataset, language='en', use_access=True, bart_model='bart.large')
    kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
    kwargs['train_kwargs']['update_freq'] = 16  # Lower this number to prevent OOM
    kwargs['train_kwargs']['batch_size'] = 16  # Lower this number to prevent OOM
    kwargs['train_kwargs']['memory_efficient_fp16'] = True
    kwargs['train_kwargs']['max_tokens'] = 2048  # Lower this number to prevent OOM
    kwargs['train_kwargs']['max_update'] = 30000  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1648626870415/checkpoints/checkpoint_68_2200.pt'

    # max tokens per GPU = 4096 and max sentences per GPU = 64
    result = fairseq_train_and_evaluate_with_parametrization(**kwargs)

elif train_lang == 'ger_mbart':
    # TODO: tokenize corpus by hand
    dataset = 'zufuss'
    kwargs = get_mbart_kwargs(dataset=dataset, language='de', use_access=False)
    kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
    kwargs['train_kwargs']['update_freq'] = 100  # Lower this number to prevent OOM
    kwargs['train_kwargs']['batch_size'] = 16  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['max_sentences'] = 6  # Lower this number to prevent OOM
    kwargs['train_kwargs']['memory_efficient_fp16'] = True
    kwargs['train_kwargs']['max_tokens'] = 1024  # Lower this number to prevent OOM
    kwargs['train_kwargs']['max_update'] = 10000  # Lower this number to prevent OOM
    kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1651001637199/checkpoints/checkpoint_118_1400.pt'
    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1649844244766/checkpoints/checkpoint_6_100.pt'

    # max tokens per GPU = 4096 and max sentences per GPU = 64
    result = fairseq_train_and_evaluate_with_parametrization(**kwargs)


elif train_lang == 'ger_mbart_middle':
    # TODO: tokenize corpus by hand
    dataset = 'uts_middle_de_query-1995d8694b2dac908ccc80b3f902cfea_db-1995d8694b2dac908ccc80b3f902cfea_topk-8_nprobe-16_density-0.6_distance-0.05_filter_ne-False_levenshtein-0.2_simplicity-0.0'
    kwargs = get_mbart_kwargs(dataset=dataset, language='de', use_access=False)
    kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
    kwargs['train_kwargs']['update_freq'] = 100  # Lower this number to prevent OOM
    kwargs['train_kwargs']['batch_size'] = 16  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['max_sentences'] = 6  # Lower this number to prevent OOM
    kwargs['train_kwargs']['memory_efficient_fp16'] = True
    kwargs['train_kwargs']['max_tokens'] = 1024  # Lower this number to prevent OOM
    kwargs['train_kwargs']['max_update'] = 10000  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1651001637199/checkpoints/checkpoint_118_1400.pt'
    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1649844244766/checkpoints/checkpoint_6_100.pt'

    # max tokens per GPU = 4096 and max sentences per GPU = 64
    result = fairseq_train_and_evaluate_with_parametrization(**kwargs)


elif train_lang == 'ger_mbart_head_middle':
    # TODO: tokenize corpus by hand
    dataset = 'uts_head_middle_paranmt'
    kwargs = get_mbart_trimmed_kwargs(dataset=dataset, language='de', use_access=True)
    kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
    kwargs['train_kwargs']['update_freq'] = 100  # Lower this number to prevent OOM
    kwargs['train_kwargs']['batch_size'] = 16  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['batch_size'] = 8  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['max_sentences'] = 6  # Lower this number to prevent OOM
    kwargs['train_kwargs']['memory_efficient_fp16'] = True
    kwargs['train_kwargs']['max_tokens'] = 1024  # Lower this number to prevent OOM
    kwargs['train_kwargs']['max_update'] = 30000  # Lower this number to prevent OOM
    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1659803361264/checkpoints/checkpoint_last.pt' # continue access from 10k to 15k
    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1660044661798/checkpoints/checkpoint_last.pt' # from 15 to 25k
    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1660134297015/checkpoints/checkpoint_last.pt'

    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1661784149491/checkpoints/checkpoint_last.pt' #continue model with 15k steps 600k samp
    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1662380038760/checkpoints/checkpoint_last.pt' # #continue model with 30k steps 600k samp - earliy ending due to no enhancement of loss

    # kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1663319157361/checkpoints/checkpoint_last.pt' # #continue model with 30k to 60k steps 800k samp
    kwargs['train_kwargs']['restore_file'] = '/home/juliogalindo/PycharmProjects/muss/muss/experiments/fairseq/local_1663573361848/checkpoints/checkpoint_last.pt' # #continue model with 60k to 90k steps 800k samp

    kwargs['train_kwargs']['save_interval_updates'] = 200
    kwargs['train_kwargs']['keep_interval_updates'] = 1
    kwargs['train_kwargs']['no_epoch_checkpoints'] = True
    kwargs['train_kwargs']['ddp_backend'] = 'no_c10d'
    kwargs['train_kwargs']['tensorboard_logdir'] = '/home/juliogalindo/PycharmProjects/muss/logs/'+dataset

    # max tokens per GPU = 4096 and max sentences per GPU = 64
    result = fairseq_train_and_evaluate_with_parametrization(**kwargs)
