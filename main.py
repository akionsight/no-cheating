import os
import torch
import scipy.io as sio
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


# Import your environment function (assuming you saved it as env.py)
from env import run_mock_env

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# 1. Initialize Model & Tokenizer once
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    dtype=torch.float16, 
    device_map="auto"
)

# /root/wtf_man_cmon/main.py

# Enable hidden states for the environment capture
model.generation_config.output_hidden_states = True
model.generation_config.return_dict_in_generate = True

# 2. Define the Permutations
task_instances = [1, 2, 3, 4, 5]
difficulties = ["impossible", "solvable"] ## trial 2 did not include solvable since that isn't a bottleneck I think.
conditions = ["control", "output", "transcript", "internals"]
seeds = [42, 100, 2026, 777, 999]  # 5 distinct seeds

# Create an output directory for the MAT files
os.makedirs("results_32b", exist_ok=True)

total_runs = len(task_instances) * len(difficulties) * len(conditions) * len(seeds)
current_run = 0

print(f"Starting rollout loop for {total_runs} permutations...")

print(model.device)

# rec = run_mock_env(model=model, tokenizer=tokenizer, difficulty="impossible", condition="control", seed=42)
# print(rec["transcript"]); print("VIOLATED:", rec["violated"])

# 3. The Execution Loop
for task_id in task_instances:
    for diff in difficulties:
        for cond in conditions:
            for seed in seeds:
                current_run += 1
                run_name = f"task{task_id}_{diff}_{cond}_seed{seed}"
                print(f"[{current_run}/{total_runs}] Running {run_name}...")
                
                try:
                    # Run the environment
                    # (Passing task_id just in case you expand env.py later to use it)
                    try: 
                        rec = run_mock_env(
                            model=model, 
                            tokenizer=tokenizer, 
                            difficulty=diff, 
                            condition=cond, 
                            seed=seed,
                            task_id= task_id
                        )

                    except Exception as e:
                        import traceback
                        print(f"❌ Error on {run_name}: {e}")
                        traceback.print_exc()

                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()



                    # Package the data for MATLAB
                    save_data = {
                        "acts": rec["acts"].astype("float16"),  # Compress for disk space
                        "violated": int(rec["violated"]),
                        "onset": rec["onset_token_index"] if rec["onset_token_index"] is not None else -1,
                        "onset_turn": rec["onset_turn"] if rec["onset_turn"] is not None else -1,
                        "difficulty": rec["difficulty"],
                        "condition": rec["condition"],
                        "task_id": task_id,
                        "seed": seed
                    }
                    
                    # Save binary MAT file
                    sio.savemat(f"results_32b/{run_name}.mat", save_data)
                    
                    # Save the raw transcript for the black-box baseline (Step 6)
                    with open(f"results_32b/{run_name}_transcript.txt", "w") as f:
                        f.write(rec["transcript"])
                        
                except Exception as e:
                    print(f"❌ Error on {run_name}: {e}")



print("\n🎉 Generation complete! Hand the text files to your teammate and jump into MATLAB for Gate 1.")
