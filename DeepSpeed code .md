# DeepSpeed code 

# [https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training)


1.<SOS>、<BOS>、<GO>：代表一个序列的开始。

2.<EOS>：代表一个序列的结束，作为判断终止的标签。

3.<MASK>：用于遮盖句子中的一些单词。

4.<UNK>：未知字符，代表词典中没有的词。

5.<SEP>: 用于分隔两个输入句子，例如输入句子 A 和 B，要在句子 A，B 后面增加 <SEP> 标志。

6.<CLS> ：放在句子的首位，表示句子的开始，就是classification的意思，通常会在bert等模型出现。

7.<PAD>：补全字符，例如要将句子处理为特定的长度，我们就要在句子前后补<PAD>。



# deepspeedchat

## data

sft\_9000\_data.py

## model

### model\_utils.py

```text/x-python
def create_hf_model 
	model_config = GPTPanguConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
  model.config.end_token_id = tokenizer.eos_token_id
  model.config.pad_token_id = tokenizer.pad_token_id
	return model
```

【原来】create\_hf\_model(**AutoModel**, model\_name\_or\_path, tokenizer,  ds\_config, rlhf\_training, disable\_dropout) 

【现在】create\_hf\_model(**GPTPanguModel**, model\_name\_or\_path, tokenizer,

                                   ds\_config, rlhf\_training, disable\_dropout)

```text/x-python
def create_critic_model
	 critic_model = create_hf_model(GPTPanguModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, disable_dropout)          
   critic_model = RewardModel_K(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning)        
 	 if rlhf_training:
   	for name, param in critic_model.named_parameters():
   		if name == 'v_head.weight':
      	param.data = torch.load(os.path.join(model_name_or_path, 'pytorch_model.bin'), map_location='cpu')["v_head.weight"]
```

### reward\_model.py

### reward\_model\_k.py

```text/x-python
B, k, max_length = input_ids.shape
input_ids = input_ids.reshape(-1, input_ids.shape[-1])
attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
input_ids = input_ids.to(torch.cuda.current_device())
attention_mask = attention_mask.to(torch.cuda.current_device())
        
#print("rewards",rewards.shape)
rewards = rewards.reshape(B, k, -1)
input_ids = input_ids.reshape(B, k, -1)

def forward()
	return {loss, 
          chosen_mean_scores,
          rejected_mean_scores}

def forward_value()
	return {
    values,
    chosen_end_scores.
  }
```

## models

gptpangu

## module

lora

# deepspeed-step 1-supervised finetuning

# deepspeed-step 2-reward model

##  training\_scripts/run\_sft9000\_RM.sh 

地址： /mnt/nasdata/share/zhaozhiruo/DeepSpeedExamples-chatgpt/applications/DeepSpeedChat/training/step2\_reward\_model\_finetuning/training\_scripts/run\_sft9000\_RM.sh 

```plaintext
deepspeed  train_pangu_RM.py \
   --train_data /mnt/nasdata/data/zhangxuechen/5_13_use_data/diedai_0/RM_train_use_data/diedai_1_RM_train_v04_28.jsonl \
   --val_data /mnt/nasdata/data/zhangxuechen/5_13_use_data/diedai_0/RM_train_use_data/diedai_1_RM_val_v04_28.jsonl \
   --model_name_or_path  /home/bml/storage/mnt/v-f9whl2maze7ifc0b/org/chengzhen/test_models/pangu_2.6B_ckpt0412 \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --weight_decay 0.1 \
   --num_train_epochs 20 \
   --disable_dropout \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT
   # &> $OUTPUT/RM_lasttoken_training.log
```

model: /mnt/nasdata/data/zhangxuechen/pangu\_2.6B\_ckpt0508

## train\_pangu\_RM.py

1.  def parse\_args( ):
    

##传参

2.  def main( )
    

##分布式训练，**args.local\_rank：分布式训练自动分配的id号**

```plaintext
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
    
    if torch.distributed.get_rank() == 0:
        writer = iSummaryWriter(log_path=args.iwriter_path, log_name='pangu2.6B-RM-sortout')

    args.global_rank = torch.distributed.get_rank()
```

##分词器，**为什么要改成这个**？

【原来】

```plaintext
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,fast_tokenizer=True)
tokenizer.pad_token = tokenizer.eos_token
```

调用DeepSpeed中的AutoTokenizer.from\_pretrained可以自动解析args.model\_name\_or\_path对应的分词器；

第二行，需要保持每个batch的长度一致，以满足pytorch定长tensor的。

【现在】

```plaintext
tokenizer = GPTPanguTokenizer.from_pretrained(args.model_name_or_path)
```

地址：DeepSpeedChat/training/deepspeedchat/models/gptpangu/tokenization\_gptpangu.py

```plaintext
from transformers.tokenization_utils import PreTrainedTokenizer
class GPTPanguTokenizer(PreTrainedTokenizer):
```

##Reward model，**create\_critic\_model**

```plaintext
from deepspeedchat.model.model_utils import create_critic_model

rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   disable_dropout=args.disable_dropout)
    
if args.lora_dim > 0:
	rm_model = convert_linear_layer_to_lora(rm_model,
                                          args.lora_module_name,
                                          args.lora_dim)
	if args.only_optimize_lora:
		rm_model = only_optimize_lora_parameters(rm_model)
```

地址：/mnt/nasdata/share/zhaozhiruo/DeepSpeedExamples-chatgpt/applications/DeepSpeedChat/training/deepspeedchat/model/model\_utils.py

【原来】**RewardModel.py**

【现在】**RewardModel\_K.py**

##Dataset

【原来】**create\_prompt\_dataset， from deepspeedchat.data.data\_utils**

```plaintext
  train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)
```

【现在】**RewardDataset， RewardSFTDataset， from deepspeedchat.data.reward\_dataset**

```plaintext
    train_dataset = RewardDataset(args.train_data, tokenizer, args.max_seq_len,mode="train")
    eval_dataset = RewardDataset(args.val_data, tokenizer, args.max_seq_len, mode = 'test')
    #eval_dataset = RewardSFTDataset(args.val_data, tokenizer, args.max_seq_len, mode = 'test')
```

##DataLoaders

【原来】**DataCollatorReward，  from deepspeedchat.data.data\_utils**

```plaintext
data_collator = DataCollatorReward()
```

【现在】**DataCollatorReward\_K， from deepspeedchat.data.data\_utils**

```plaintext
 data_collator = DataCollatorReward_K(tokenizer)
```

##**evaluation\_reward**

用于评价模型训练结果，在模型训练前，训练过程中和训练结束后各自计算reward\_score, reject\_scores, acc, val\_loss，进行比较。

```text/x-python
    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        reject_scores = 0
        val_loss = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            reject_scores += outputs["rejected_mean_scores"].mean().float()
            scores += outputs["chosen_mean_scores"].mean().float()
            val_loss += outputs["loss"].mean().float()
            acc = correct_predictions / total_predictions
            scores = scores / (step + 1)
            reject_scores = reject_scores / (step + 1)
            val_loss = val_loss / (step + 1)
        try:
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
            reject_scores = get_all_reduce_mean(reject_scores).item()
            val_loss = get_all_reduce_mean(val_loss).item()
        except:
            pass
        return scores, reject_scores, acc, val_loss
```

# deepspeed-step 3-rlhf

## training\_scripts/run\_sft9000\_RL.sh

地址：/mnt/nasdata/share/zhaozhiruo/DeepSpeedExamples-chatgpt/applications/DeepSpeedChat/training/step3\_rlhf\_finetuning/training\_scripts/run\_sft9000\_RL.sh

```text/x-python
ACTOR_ZERO_STAGE="--actor_zero_stage 3"
CRITIC_ZERO_STAGE="--critic_zero_stage 3"
ACTOR_MODEL_PATH=/home/bml/storage/mnt/v-f9whl2maze7ifc0b/org/chengzhen/test_models/pangu_2.6B_ckpt0402/pangu_2.6B_ckpt0402/
CRITIC_MODEL_PATH=/home/bml/storage/mnt/v-f9whl2maze7ifc0b/org/yinwenbo/yxm_RM/deepspeed/pangu2.6Blora_shortout_k2_lasttoken/20/
```
```text/x-python
deepspeed --master_port 12346 train_pangu_RL.py \
   --train_data /mnt/nasdata/share/yxm/data/RLdata/数据切分1_0309/RL_train.json \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 4 \
   --per_device_mini_train_batch_size 4 \
   --generation_batch_numbers 50 \
   --ppo_epochs 100 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 96 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1000 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   ${ACTOR_ZERO_STAGE} \
   ${CRITIC_ZERO_STAGE} \
   --actor_lora_dim 128 \
   --only_optimize_lora \
   --enable_hybrid_engine \
   --enable_ema \
   --inference_tp_size 1 \
   --actor_lora_module_name transformer.h. \
   --output_dir $OUTPUT
```

## 2. train\_pangu\_RL.py

**engine** = DeepSpeedRLHFEngine(actor\_model\_name\_or\_path=actor\_model\_name\_or\_path,  
                                                        critic\_model\_name\_or\_path=critic\_model\_name\_or\_path,  
                                                        tokenizer=tokenizer,  
                                                        args=args)  
**trainer** = DeepSpeedPPOTrainer(engine=engine, args=args)

1.  def parse\_args( ):
    

1.  def create\_datasets(args, tokenizer, train\_phase=3): 
    

return **prompt\_train\_dataloader, unsupervised\_train\_dataloader, num\_total\_iters**

 **prompt\_train\_dataloader** 和  **unsupervised\_train\_dataloader**分别用于训练啥？

2.  def main( )
    

```plaintext
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
    
    if torch.distributed.get_rank() == 0:
        writer = iSummaryWriter(log_path=args.iwriter_path, log_name='pangu2.6B-RM-sortout')

    args.global_rank = torch.distributed.get_rank()
```

##为什么actor model的batch size需要double?

```text/x-python
unsupervised_training_enabled = args.unsupervised_dataset_name is not None #and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps
```

##exp\_mini\_dataset和exp\_mini\_dataset 

```text/x-python
 # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                   args.per_device_mini_train_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                     args.per_device_mini_train_batch_size)
```

##PPO训练

```text/x-python
                for ppo_ep in range(args.ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        actor_loss, critic_loss, rewards_kl = trainer.train_rlhf(exp_data)
                        critic_loss += actor_loss.item()
                        actor_loss += critic_loss.item()
                        average_rewards_kl += rewards_kl
                        average_reward += exp_data["rewards"].mean()

                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            unsuper_loss += unsup_loss.item()

                        inner_iter += 1
                        if args.enable_ema:
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)
```

## 3. rlhf\_engine.py

##加载、Actor/Critic Model，Reference, Ema等trick

```text/x-python
class DeepSpeedRLHFEngine():
 def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                 tokenizer, args, num_total_iters):
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer

        self.actor = self._init_gptpangu_actor(
            actor_model_name_or_path=actor_model_name_or_path)
        self.ref = self._init_ref(
            actor_model_name_or_path=actor_model_name_or_path)
        self.actor_ema = None
        if self.args.enable_ema:
            self.actor_ema = self._init_ema(
                actor_model_name_or_path=actor_model_name_or_path)

        self.critic = self._init_critic(
            critic_model_name_or_path=critic_model_name_or_path)
        self.reward = self._init_reward(
            critic_model_name_or_path=critic_model_name_or_path)
        
        if self.args.critic_gradient_checkpointing:
            self.critic.gradient_checkpointing_enable()
```

## 3. ppo\_trainer.py

##class DeepSpeedPPOTrainer():

```text/x-python
def generate_experience(self, prompts):
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]
        logits = output.logits
        logits_ref = output_ref.logits
        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask
        }
```

```text/x-python
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        rewards_kl = 0
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]
            rewards_kl += rewards[j, start:ends[j]][-1]

        return rewards, rewards_kl/batch_size
```

##class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

```text/x-python
def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = self.actor_model(input_ids,attention_mask=attention_mask , use_cache=False)
        lm_logits = outputs.logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ptx_loss = self.ptx_loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1).long())
        ptx_loss = (attention_mask[...,1:].reshape(-1) * ptx_loss).mean()
        #loss = outputs.loss
        self.actor_model.backward(unsup_coef * ptx_loss)
        self.actor_model.step()

        return ptx_loss
```
