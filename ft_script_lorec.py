import subprocess
import argparse
import os
import time

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, nargs='+')
parser.add_argument("--train_size", type=int, nargs='+')
parser.add_argument("--train_type", type=str, nargs='+')
parser.add_argument("--test_type", type=str, nargs='+')
parser.add_argument("--dataset", type=str)
parser.add_argument("--K", type=int, nargs='+')
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--bs", type=int, default=256)
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--debugging", type=bool, default=False)
parser.add_argument("--mode", type=str, default='origin')
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--model_name", type=str, default='vicuna-13b')
parser.add_argument("--layers", type=str, default='[q, v, up, down]')
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--enable_softmax", action="store_true")
parser.add_argument("--enable_sqrt_after_softmax", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--ret", action="store_true")
parser.add_argument("--load_rella", action="store_true")
parser.add_argument("--all_init_one", action="store_true")
parser.add_argument("--test_range", type=str, default='all')
parser.add_argument("--head", type=str, default='ctr2')
parser.add_argument("--init_method", type=str, default='kaiming')
parser.add_argument("--lora_assemble_num", type=int, default=8)
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--routing_mode", type=str, default='soft')
parser.add_argument("--which_in_gate", type=str, default='None')
parser.add_argument("--ctr_dropout", type=float, default=0.1)
parser.add_argument("--agg_pt", type=str, default='w')
parser.add_argument("--ctr_out_layer", type=str, default='final')
parser.add_argument("--ctr_K", type=int, default=30)

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.makedirs(f"{args.dataset}_logs", exist_ok=True)
os.makedirs(f"{args.dataset}_lora-Vicuna", exist_ok=True)
print(os.environ["CUDA_VISIBLE_DEVICES"])
CUDA_NUM = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

print(CUDA_NUM)
for test_type in args.test_type:
    for train_type in args.train_type:
        for lr in args.lr:
            for train_size in args.train_size:
                for K in args.K:
                    fp = f"_Dataset_{args.dataset}_CUDA_NUM_{CUDA_NUM}_test_range_{args.test_range}_{args.enable_softmax}_{args.enable_sqrt_after_softmax}_layers_{args.layers}_lr_{lr}_shot_{train_size}_{train_type}_{test_type}_K_{K}_ctr_K_{args.ctr_K}_{args.epochs}_bs{args.bs}_wd_{args.weight_decay}_model_{args.model_name}_mode_{args.mode}_lora_r_{args.lora_r}_lora_num_{args.lora_assemble_num}__temp_{args.temperature}_rout_{args.routing_mode}_ret_{args.ret}_lora_drpt_{args.lora_dropout}"
                    run_py = "python finetune_lorec.py " if CUDA_NUM == 1 else f"torchrun --nproc_per_node={CUDA_NUM} --master_port=12345 finetune_lorec.py "
                    command = \
                        run_py + \
                        f"--lr {lr} "\
                        f"--dataset {args.dataset} "\
                        f"--train_size {train_size} "\
                        f"--train_type {train_type} "\
                        f"--test_range {args.test_range} "\
                        f"--test_type {test_type} "\
                        f"--K {K} "\
                        f"--epochs {args.epochs} "\
                        f"--total_batch_size {args.bs} "\
                        f"--debugging {args.debugging} "\
                        f"--output_path {args.dataset}_lorec/{fp} "\
                        f"--mode {args.mode} "\
                        f"--head {args.head} "\
                        f"--init_method {args.init_method} "\
                        f"--weight_decay {args.weight_decay} "\
                        f"--model_name {args.model_name} "\
                        f"--layers {args.layers} "\
                        f"--lora_dropout {args.lora_dropout} "\
                        f"--ctr_dropout {args.ctr_dropout} "\
                        f"--agg_pt {args.agg_pt} "\
                        f"--lora_assemble_num {args.lora_assemble_num} "\
                        f"--lora_r {args.lora_r} "\
                        f"--ctr_out_layer {args.ctr_out_layer} "\
                        f"--ctr_K {args.ctr_K} "\
                        f"--temperature {args.temperature} "\
                        f"--which_in_gate {args.which_in_gate} "\
                        f"{'--enable_softmax' if args.enable_softmax else ''} "\
                        f"{'--enable_sqrt_after_softmax' if args.enable_sqrt_after_softmax else ''} "\
                        f"{'--resume' if args.resume else ''} "\
                        f"{'--ret' if args.ret else ''} "\
                        f"{'--load_rella' if args.load_rella else ''} "\
                        f"{'--all_init_one' if args.all_init_one else ''} "\
                        # f">> {args.dataset}_lorec_logs/More_lnrelu_{fp}.txt"
                    print(command)
                    subprocess.run(command, shell=True)



print("time")
print(time.time()- t1)
print((time.time()-t1)/3600)
