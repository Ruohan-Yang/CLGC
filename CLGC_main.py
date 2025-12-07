from support_main import *
from src.Load import *


def cal_CLGC(results_dict, log_path):
    for metric in ['auc', 'ap', 'aupr']:
        results = [r[metric] for r in results_dict]
        print('\nMetric:', metric, ' Results:', results)
        N = len(results) // 2
        sharedresult = results[:N]
        selfresult = results[N:]
        total = 0
        for i in range(N):
            trans_i = min(1, sharedresult[i] / selfresult[i])
            total += trans_i
            # print(trans_i)
        clec_N = total / N
        value = round(clec_N, 6)
        print("CLGC score:", value)

        with open(log_path, 'a', encoding='utf-8') as log:
            results_str = " ".join(str(x) for x in results)
            write_infor = f"Metric: {metric}. Results: {results_str}"
            log.write(write_infor + '\n')
            write_infor = f"CLGC score: {value}"
            log.write(write_infor + '\n')

    return clec_N

def generate_combinations(Eval_layers):

    # Eval_layers = ['Aarhus_1', 'small_world']
    # [{'train': ['Aarhus_1', 'small_world'], 'test': ['Aarhus_1']}, {'train': ['Aarhus_1', 'small_world'], 'test': ['small_world']},
    #  {'train': ['Aarhus_1', 'Aarhus_1'], 'test': ['Aarhus_1']}, {'train': ['small_world', 'small_world'], 'test': ['small_world']}]

    # Eval_layers = ['Aarhus_1', 'Aarhus_2', 'Aarhus_3']
    # [{'train': ['Aarhus_1', 'Aarhus_2', 'Aarhus_3'], 'test': ['Aarhus_1']}, {'train': ['Aarhus_1', 'Aarhus_2', 'Aarhus_3'], 'test': ['Aarhus_2']},
    #  {'train': ['Aarhus_1', 'Aarhus_2', 'Aarhus_3'], 'test': ['Aarhus_3']}, {'train': ['Aarhus_1', 'Aarhus_1', 'Aarhus_1'], 'test': ['Aarhus_1']},
    #  {'train': ['Aarhus_2', 'Aarhus_2', 'Aarhus_2'], 'test': ['Aarhus_2']}, {'train': ['Aarhus_3', 'Aarhus_3', 'Aarhus_3'], 'test': ['Aarhus_3']}]

    combos = []
    train_all = Eval_layers.copy()
    for test_layer in train_all:
        combos.append({
            'train': train_all,
            'test': [test_layer]
        })

    for layer in Eval_layers:
        train_repeat = [layer] * len(Eval_layers)
        combos.append({
            'train': train_repeat,
            'test': [layer]
        })
    return combos

if __name__ == '__main__':

    args = get_args()
    setup_seed(seed=args.set_seed)
    log_path = './CLGC_log/'
    os.makedirs(log_path, exist_ok=True)
    save_path = './save/'
    os.makedirs(save_path, exist_ok=True)

    Eval_layers = args.Eval_layers
    print(Eval_layers)
    assert len(Eval_layers) >= 2, "CLGC is used to compute cross-layer generation consistency between two or more networks."
    Eval_layers_str = "-".join(Eval_layers)
    args.log = log_path + 'CLGC-' + Eval_layers_str + '.txt'
    print(args)

    begin = datetime.datetime.now()
    print('Start time ', begin)
    time = str(begin.year) + '-' + str(begin.month) + '-' + str(begin.day) + '-' + str(begin.hour) + '-' + str(
        begin.minute) + '-' + str(begin.second)
    log = open(args.log, 'a', encoding='utf-8')
    write_infor = '\nStart time: ' + time + '\n'
    log.write(write_infor)
    write_infor = ', '.join([f"{k}: {v}" for k, v in vars(args).items()]) + '\n'
    log.write(write_infor)
    log.close()

    network_numbers = len(Eval_layers)
    combinations = generate_combinations(Eval_layers)
    print('case combinations:', combinations)

    network_infor = pro_data_CLGC(Eval_layers, args.set_seed)

    results = []
    gcn_data = []
    for case in combinations:
        train_networks = case['train']
        eval_name = case['test'][0]

        train_infor = []
        for name in train_networks:
            layer = network_infor[name]['lay_index']
            gcn_data.append(network_infor[layer]["layerwise_gcn_data"])
            layer_train_infor = network_infor[layer]["train"]
            train_infor.append(layer_train_infor)
        train_infor = np.vstack(train_infor)
        np.random.shuffle(train_infor)
        layer = network_infor[eval_name]['lay_index']
        valid_infor = network_infor[layer]["valid"]
        test_infor = network_infor[layer]["test"]
        print("train counter: ", sorted(Counter(train_infor[:, 3]).items()))
        print("valid counter: ", sorted(Counter(valid_infor[:, 3]).items()))
        print("test counter: ", sorted(Counter(test_infor[:, 3]).items()))
        print('-----------------------------------')
        train_loader = get_loader(train_infor, args.batch_size)
        valid_loader = get_loader(valid_infor, args.batch_size)
        test_loader = get_loader(test_infor, args.batch_size)

        log = open(args.log, 'a', encoding='utf-8')
        train_str = ",".join(train_networks)
        write_infor = f"Train_networks: {train_str}\nEval_networks: {eval_name}"
        print(write_infor)
        log.write(write_infor + '\n')
        log.close()

        model = Model_Net(embedding_dim=args.dim, layer_number=network_numbers, gcn_data=gcn_data, gcn_type=args.gcn_type, gcn_layer=args.gcn_layer)
        if torch.cuda.is_available():
            model = model.cuda()
            for i in range(network_numbers):
                gcn_data[i].x = gcn_data[i].x.cuda()
                gcn_data[i].edge_index = gcn_data[i].edge_index.cuda()

        test_results = run_model(train_loader, valid_loader, test_loader, model, args, log_mode='CLGC')
        results.append(test_results)

    cal_CLGC(results, args.log)
    end = datetime.datetime.now()
    print('End time ', end)
    print('Run time ', end - begin)
