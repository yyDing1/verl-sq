# Eval Data Process
# python3 -m examples.data_preprocess.r1_bench \
#     --local_dir /workspace/datasets/livecodebench \
#     --tasks livecodebench

# Generation
# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=1 \
#     trainer.n_gpus_per_node=8 \
#     data.path=/workspace/datasets/livecodebench/test.parquet \
#     data.prompt_key=prompt \
#     data.batch_size=1024 \
#     data.n_samples=1 \
#     data.output_path=/workspace/datasets/livecodebench/test-output-1.parquet \
#     model.path=/workspace/hf_models/DeepSeek-R1-Distill-Qwen-1.5B \
#     rollout.temperature=0.6 \
#     rollout.top_p=0.95 \
#     rollout.prompt_length=1024 \
#     rollout.response_length=32768 \
#     rollout.tensor_model_parallel_size=1 \
#     rollout.gpu_memory_utilization=0.9 \
#     rollout.max_num_batched_tokens=65536

# Evaluation
python3 -m verl.trainer.main_eval \
    data.path=/workspace/datasets/livecodebench/test-output-1.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
