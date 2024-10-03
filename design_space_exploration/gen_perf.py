import json, re
from hardware_model.compute_module import (
    VectorUnit,
    SystolicArray,
    Core,
    ComputeModule,
    overhead_dict,
)
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule
from hardware_model.device import Device
from hardware_model.interconnect import LinkModule, InterConnectModule, TopologyType
from hardware_model.system import System
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
from math import ceil
from design_space_exploration.dse import read_architecture_template, template_to_system
import random
import os
import csv
from multiprocessing import Pool

def predict_latency(arch_path, reqest_desription):
    """
    predict the performance given hardware decription, LLM model description

    Parameters:
    arch_path (str): hardware config path
    reqest_desription (map): desrible 

    Returns:
    prompt time: 
    token time:
    """
    arch_specs = read_architecture_template(arch_path)
    system = template_to_system(arch_specs)
   
    model_auto_regression = TransformerBlockAutoRegressionTP(
            d_model = reqest_desription['dim'],
            n_heads = reqest_desription['head'],
            device_count= arch_specs['device_count'],
            data_type= data_type_dict["fp16"],
        )
    _ = model_auto_regression(Tensor([1, 1, model_auto_regression.d_model],data_type_dict["fp16"]), (reqest_desription['prompt_size'] + reqest_desription['token_size']) * reqest_desription['batch_size'])
    # print('----------simulating token phase-----------')
    token_time = model_auto_regression.compile_and_simulate(system, 'heuristic-GPU') * 1000 * reqest_desription['layers']
    model_init = TransformerBlockInitComputationTP(
            d_model = reqest_desription['dim'],
            n_heads = reqest_desription['head'],
            device_count= arch_specs['device_count'],
            data_type= data_type_dict["fp16"],
        )
    _ = model_init(Tensor([reqest_desription['batch_size'], reqest_desription['prompt_size'], model_init.d_model], data_type_dict["fp16"]))
    # print('----------simulating prompt phase-----------')
    prompt_time = model_init.compile_and_simulate(system, 'heuristic-GPU') * 1000 * reqest_desription['layers']
    return prompt_time, token_time

def predict_area(arch_specs):
    total_area_mm2=calc_compute_chiplet_area_mm2(arch_specs)+calc_io_die_area_mm2(arch_specs)
    return total_area_mm2 * arch_specs['device_count']


def get_unique_combinations(file_path):
    unique_combinations = set()
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Create a tuple of the keys you want
            if row['model'] == 'llama2-70b':
                continue
            combination = (int(row['batch_size']), int(row['prompt_size']), int(row['token_size']))
            unique_combinations.add(combination)
    return unique_combinations
def gen_unique_combinations():
    unique_combinations = set()
    for batch_size in [1, 2, 4, 8, 16]:
        for prompt_size in [128, 256, 512, 1024, 2048, 4096]:
            for total_size in [256, 512, 1024, 2048, 4096, 90]:
                token_size = prompt_size 
                if batch_size * prompt_size > 2 * 4096:
                    continue
                else:
                    combination = (batch_size, prompt_size, token_size)
                    unique_combinations.add(combination)
    return unique_combinations


def process_sample(args):
    sample_index, unique_combinations = args
    config_name = f'hardware_{sample_index}.json'
    perf_name = f'hardware_{sample_index}_perf.csv'
    log_name = f'hardware_{sample_index}_log.txt'
    hardware_config_path = os.path.join('/data/hdliu21/DSE4LLMServing/LLMCompass/hardware_config', config_name)
    hardware_perf_path = os.path.join('/data/hdliu21/DSE4LLMServing/LLMCompass/hardware_perf', perf_name)
    log_path = os.path.join('/data/hdliu21/DSE4LLMServing/LLMCompass/logs', log_name)
    header = ['model', 'hardware', 'batch_size', 'prompt_size', 'token_size', 'prompt_time', 'token_time']
    
    reqest_desription = {
        'dim': 12288,
        'head': 96,
        'layers': 96
    }
    # reqest_desription = {
    #     'dim': 14336,
    #     'head': 112,
    #     'layers': 70
    # }
    
    with open(hardware_perf_path, 'w', newline='') as file, open(log_path, 'w') as log_file:
        writer = csv.writer(file)
        writer.writerow(header)
        file.flush()
        print(f"Processing sample {sample_index}", file=log_file, flush=True)
        for combination in unique_combinations:
            reqest_desription['batch_size'] = combination[0]
            reqest_desription['prompt_size'] = combination[1]
            reqest_desription['token_size'] = combination[2]
            try:
                print(f"Start predict combination {combination}", file=log_file, flush=True)
                prompt_time, token_time = predict_latency(hardware_config_path, reqest_desription)
                writer.writerow(['GPT-3', f'hardware-{sample_index}', combination[0], combination[1], combination[2], prompt_time, token_time])
                file.flush()
                print(f"Processed combination {combination} with prompt_time: {prompt_time}, token_time: {token_time}", file=log_file, flush=True)
            except Exception as e:
                print(f"An error occurred for combination {combination}: {e}", file=log_file, flush=True)

if __name__ == '__main__':
    unique_combinations = get_unique_combinations('/data/hdliu21/workspace/splitwise-sim/data/perf_model.csv')
    #unique_combinations = gen_unique_combinations()
    random.seed(0)
    samples = random.sample(range(10000), 1000)
    
    max_processes = 40  # Specify the maximum number of processes
    with Pool(processes=max_processes) as pool:
        pool.map(process_sample, [(sample_index, unique_combinations) for sample_index in samples])
        print('All tasks have been completed.')
        
# if __name__ == '__main__':
#     unique_combinations = get_unique_combinations('/data/hdliu21/workspace/splitwise-sim/data/perf_model.csv')
#     print('number of combinations', len(unique_combinations))
#     for combination in unique_combinations:
#         print(combination)

    

    # arch_specs = read_architecture_template('/data/hdliu21/workspace/DSE4LLM/LLMCompass/llmcompass/configs/GA100.json')
    # model_description = {}
    # model_description['dim'] = 12288
    # model_description['head'] = 96
    # model_description['batch_size'] = 8
    # model_description['input_length'] = 512
    # model_description['output_length'] = 128
    # prompt_time = predict_latency(arch_specs=arch_specs, model_description = model_description, phase = 'prompt')
    # print('prompt time is ', prompt_time * 1000 * 70)
    # token_time = predict_latency(arch_specs=arch_specs, model_description = model_description, phase = 'token')
    # print('token time is ', token_time * 1000 * 96)

# if __name__ == '__main__':
#     arch_specs = read_architecture_template("configs/template.json")
#     batch_size = 8
#     device_count = 8
#     input_seq_length = 2048
#     output_seq_length = 1024
#     model_init = TransformerBlockInitComputationTP(
#                 d_model=12288,
#                 n_heads=96,
#                 device_count=device_count,
#                 data_type=data_type_dict["fp16"],
#             )
#     model_auto_regression = TransformerBlockAutoRegressionTP(
#                 d_model=12288,
#                 n_heads=96,
#                 device_count=device_count,
#                 data_type=data_type_dict["fp16"],)
#     _ = model_init(Tensor([batch_size, input_seq_length, model_init.d_model], data_type_dict["fp16"]))
#     _ = model_auto_regression(Tensor([batch_size, 1, model_init.d_model],data_type_dict["fp16"]), input_seq_length+output_seq_length)
#     system=template_to_system(arch_specs)
#     auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(system, 'heuristic-GPU')
#     # init_latency_simulated = model_init.compile_and_simulate(system, 'heuristic-GPU')
#     print('auto_regression_latency_simulated', auto_regression_latency_simulated)
#     # print('init_latency_simulated', init_latency_simulated)