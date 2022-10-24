# Attack 
import os
import json
import argparse
import datetime
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import logger, result_visualizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/lws_config.json')
    parser.add_argument('--log_file', type=str, default='./attack_experiment_logs.txt')
    args = parser.parse_args()
    return args

def file_result_visualizer(result, file_handle):
    stream_writer = file_handle.write
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    left = []
    right = []
    for key, val in result.items():
        left.append(" " + key + ": ")
        if isinstance(val, bool):
            right.append(" yes" if val else " no")
        elif isinstance(val, int):
            right.append(" %d" % val)
        elif isinstance(val, float):
            right.append(" %.5g" % val)
        else:
            right.append(" %s" % val)
        right[-1] += " "

    max_left = max(list(map(len, left)))
    max_right = max(list(map(len, right)))
    if max_left + max_right + 3 > cols:
        delta = max_left + max_right + 3 - cols
        if delta % 2 == 1:
            delta -= 1
            max_left -= 1
        max_left -= delta // 2
        max_right -= delta // 2
    total = max_left + max_right + 3

    title = "Summary"
    if total - 2 < len(title):
        title = title[:total - 2]
    offtitle = ((total - len(title)) // 2) - 1
    stream_writer("+" + ("=" * (total - 2)) + "+\n")
    stream_writer("|" + " " * offtitle + title + " " * (total - 2 - offtitle - len(title)) + "|" + "\n")
    stream_writer("+" + ("=" * (total - 2)) + "+\n")
    for l, r in zip(left, right):
        l = l[:max_left]
        r = r[:max_right]
        l += " " * (max_left - len(l))
        r += " " * (max_right - len(r))
        stream_writer("|" + l + "|" + r + "|" + "\n")
    stream_writer("+" + ("=" * (total - 2)) + "+\n")

def display_results(config, results, log_file):
    poisoner = config['attacker']['poisoner']['name']
    poison_rate = config['attacker']['poisoner']['poison_rate']
    label_consistency = config['attacker']['poisoner']['label_consistency']
    label_dirty = config['attacker']['poisoner']['label_dirty']
    target_label = config['attacker']['poisoner']['target_label']
    poison_dataset = config['poison_dataset']['name']
    CACC = results['test-clean']['accuracy']
    if 'test-poison' in results.keys():
        ASR = results['test-poison']['accuracy']
    else:
        asrs = [results[k]['accuracy'] for k in results.keys() if k.split('-')[1] == 'poison']
        ASR = max(asrs)

    PPL = results["ppl"]
    GE = results["grammar"]
    USE = results["use"]

    display_result = {'poison_dataset': poison_dataset, 'poisoner': poisoner, 'poison_rate': poison_rate, 
                        'label_consistency':label_consistency, 'label_dirty':label_dirty, 'target_label': target_label,
                      "CACC" : CACC, 'ASR': ASR, "ΔPPL": PPL, "ΔGE": GE, "USE": USE}

    result_visualizer(display_result)
    with open(log_file, 'a') as log_f:
        log_f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
        file_result_visualizer(display_result, log_f)

def main(config, log_file):
    # use the Hugging Face's datasets library 
    # change the SST dataset into 2-class  
    # choose a victim classification model 
    
    # choose Syntactic attacker and initialize it with default parameters 
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    # choose SST-2 as the evaluation data  
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])


    # tmp={}
    # for key, value in poison_dataset.items():
    #     tmp[key] = value[:300]
    # poison_dataset = tmp

    # target_dataset = attacker.poison(victim, target_dataset)
    # launch attacks
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config) 
    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)
    
    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset)

    display_results(config, results, log_file)

if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    label_consistency = config['attacker']['poisoner']['label_consistency']
    label_dirty = config['attacker']['poisoner']['label_dirty']
    if label_consistency:
        config['attacker']['poisoner']['poison_setting'] = 'clean'
    elif label_dirty:
        config['attacker']['poisoner']['poison_setting'] = 'dirty'
    else:
        config['attacker']['poisoner']['poison_setting'] = 'mix'

    poisoner = config['attacker']['poisoner']['name']
    poison_setting = config['attacker']['poisoner']['poison_setting']
    poison_rate = config['attacker']['poisoner']['poison_rate']
    label_consistency = config['attacker']['poisoner']['label_consistency']
    label_dirty = config['attacker']['poisoner']['label_dirty']
    target_label = config['attacker']['poisoner']['target_label']
    poison_dataset = config['poison_dataset']['name']

    # path to a fully-poisoned dataset
    poison_data_basepath = os.path.join('poison_data', 
                            config["poison_dataset"]["name"], str(target_label), poisoner)
    config['attacker']['poisoner']['poison_data_basepath'] = poison_data_basepath
    # path to a partly-poisoned dataset
    config['attacker']['poisoner']['poisoned_data_path'] = os.path.join(poison_data_basepath, poison_setting, str(poison_rate))
    
    load = config['attacker']['poisoner']['load']
    clean_data_basepath = config['attacker']['poisoner']['poison_data_basepath']
    config['target_dataset']['load'] = load
    config['target_dataset']['clean_data_basepath'] = os.path.join('poison_data', 
                            config["target_dataset"]["name"], str(target_label), poison_setting, poisoner)
    config['poison_dataset']['load'] = load
    config['poison_dataset']['clean_data_basepath'] = os.path.join('poison_data', 
                            config["poison_dataset"]["name"], str(target_label), poison_setting, poisoner)

    main(config, args.log_file)
