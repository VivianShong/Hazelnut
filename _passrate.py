import sys, torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import load_dataset
from rewards import score_completion

source = sys.argv[1] if len(sys.argv) > 1 else "mbpp"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 384

mp = '/opt/Hazelnut/models/qwen3.5-2b-Base'
tok = AutoTokenizer.from_pretrained(mp); tok.padding_side = 'left'
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16, device_map='cuda'); model.eval()

probs = load_dataset('train', source=source,
                     limit=N, difficulty=('introductory' if source == 'apps' else None))
N = len(probs)
print(f'probing {N} {source} problems, 1 sample each, {max_new} tok')
chats = [[{'role': 'user', 'content': p['prompt']}] for p in probs]
enc = tok.apply_chat_template(chats, add_generation_prompt=True, return_tensors='pt',
                              return_dict=True, padding=True, truncation=True, max_length=1024).to('cuda')
plen = enc['input_ids'].shape[1]
t0 = time.time()
with torch.no_grad():
    out = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95,
                         max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
print(f'gen {N}x{max_new} in {time.time()-t0:.0f}s')
comp = tok.batch_decode(out[:, plen:], skip_special_tokens=True)
nc = npass = 0
fracs = []
for i, c in enumerate(comp):
    bd = score_completion(c, probs[i], max_cases=6)
    nc += bd.compiles; npass += (1 if bd.correctness > 0 else 0); fracs.append(bd.correctness)
    print(f'  p{i}: compile={bd.compiles:.0f} correct_frac={bd.correctness:.2f} ({bd.n_passed}/{bd.n_tests})')
print(f'RESULT compile_rate={nc/N:.2f} pass_any_rate={npass/N:.2f} mean_correct_frac={sum(fracs)/N:.2f}')
