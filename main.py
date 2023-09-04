import json
import os
import pandas as pd
import torch


def read_json_file(file_path):
    # Read json file
    all_data = []
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                all_data.extend(data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {file_path}")

    return all_data


def policy_buy(data):
    sorted_data = sorted(data, key=lambda x: x[0])
    latest_data = sorted_data[-1]
    policy_buy_data = latest_data[2][0]

    return policy_buy_data


def policy_sell(data):
    sorted_data = sorted(data, key=lambda x: x[0])
    latest_data = sorted_data[-1]
    policy_sell_data = latest_data[2][1]

    return policy_sell_data


def rank_policy_buy(output_name):
    folder_path = './../ai-model/output/'
    output_path = os.path.join(folder_path, output_name)
    files = [f for f in os.listdir(output_path) if f.endswith('.json') and f != 'params.json']
    ranking = []

    for file in files:
        file_path = os.path.join(output_path, file)  # Corrected the file_path
        data = read_json_file(file_path)
        pb = policy_buy(data)
        code = file[5:11]
        code = f"'{code:0>6}"
        ranking.append([code, pb])  # Fixed append statement

    df = pd.DataFrame(ranking, columns=['종목 코드', '매수 확률'])
    df = df.sort_values(by=['매수 확률'], ascending=False)
    df['순위'] = df['매수 확률'].rank(ascending=False).astype(int)
    df['종목 코드'] = df['종목 코드'].astype(str)
    df = df.set_index('순위')

    sector = output_name.split('_')[0]
    update_date = output_name.split('_')[2]
    learner = output_name.split('_')[3]
    net = output_name.split('_')[4]
    df.to_csv(f'./rank_policy_buy_{sector}_{update_date}_{learner}_{net}.csv')


def rank_policy_sell(output_name):
    folder_path = './../ai-model/output/'
    output_path = os.path.join(folder_path, output_name)
    files = [f for f in os.listdir(output_path) if f.endswith('.json') and f != 'params.json']
    ranking = []

    for file in files:
        file_path = os.path.join(output_path, file)  # Corrected the file_path
        data = read_json_file(file_path)
        ps = policy_sell(data)
        code = file[5:11]
        code = f"'{code:0>6}"
        ranking.append([code, ps])  # Fixed append statement

    df = pd.DataFrame(ranking, columns=['종목 코드', '매도 확률'])
    df = df.sort_values(by=['매도 확률'], ascending=False)
    df['순위'] = df['매도 확률'].rank(ascending=False).astype(int)
    df['종목 코드'] = df['종목 코드'].astype(str)
    df = df.set_index('순위')

    sector = output_name.split('_')[0]
    update_date = output_name.split('_')[2]
    learner = output_name.split('_')[3]
    net = output_name.split('_')[4]
    df.to_csv(f'./rank_policy_sell_{sector}_{update_date}_{learner}_{net}.csv')


for function in [rank_policy_buy, rank_policy_sell]:
    for sector in ['G4050', 'G4510', 'G4530', 'G4540', 'G5010', 'G5020', 'G5510']:
        for update_date in ['20230904']:
            for learner in ['a3c_lstm']:
                for net in ['policy', 'value']:
                    function(f'{sector}_predict_{update_date}_{learner}_{net}')

# Path: ai-model/output/G4050_predict_20230904_a3c_lstm_policy
