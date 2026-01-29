## Training

Training requires DeepSpeed as dependency:
```
pip install deepspeed
```

### Tool Retrieval Alignment
In the second stage, we train the ToolWeaver model with queries and tool tokens.
```bash
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25024 train.py \
  --model_name_or_path checkpoints/ToolWeaver-Llama-3-8B-Tool-Memorization \
  --add_virtual_tokens False \
  --flash_attention True \
  --deepspeed src/configs/ds_z2_config.json \
  --chat True \
  --template llama-3 \
  --architecture causal \
  --output_dir checkpoints/ToolWeaver-Llama-3-8B-Tool-Retriever \
  --save_strategy steps \
  --save_steps 1000 \
  --gather_weights True \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --datasets ToolWeaver_atomic_retrieval_G123.json \
  --dataset_nums 1000000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --max_length 1024 \
  --num_train_epochs 1 \
  --gradient_checkpointing False \
  --bf16 True \
  --logging_steps 1 \
  --report_to wandb \
  --run_name llama-3-8b-tool-retrieval
```

###  Tool Usage Trajectory Alignment
In the last stage, we train the ToolWeaver agent model with end-to-end trajectories. We set the maximum length to 6144, which generally needs large GPU memory. Based on our experiments, 4 GPUs each with 80GB memory are enough for this stage (Deepspeed zero 3 with offloading is used).
```bash
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25024 train.py \
  --model_name_or_path checkpoints/ToolWeaver-Llama-3-8B-Tool-Retriever \
  --add_virtual_tokens False \
  --flash_attention True \
  --deepspeed src/configs/ds_z3_offload_config.json \
  --chat True \
  --template llama-3 \
  --architecture causal \
  --output_dir checkpoints/ToolWeaver-Llama-3-8B \
  --save_strategy steps \
  --save_steps 1000 \
  --gather_weights True \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --datasets ToolWeaver_atomic_G123_dfs.json \
  --dataset_nums 10000000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --max_length 6144 \
  --num_train_epochs 1 \
  --gradient_checkpointing False \
  --bf16 True \
  --logging_steps 1 \
  --report_to wandb \
  --run_name llama-3-8b-end2end
```
