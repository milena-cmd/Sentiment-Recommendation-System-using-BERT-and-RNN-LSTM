# src/train.py

import argparse
import os

def main(model_choice):
    if model_choice == 'bert':
        from bert_model import train_bert_model
        data_file = os.path.join(os.getcwd(), 'data', 'processed', 'oula_processed.csv')
        model_save_path = os.path.join(os.getcwd(), 'results', 'bert_model')
        os.makedirs(model_save_path, exist_ok=True)
        train_bert_model(data_file, model_save_path)
    elif model_choice == 'rnn':
        from rnn_lstm_model import train_rnn_lstm_model
        data_file = os.path.join(os.getcwd(), 'data', 'processed', 'oula_processed.csv')
        model_save_path = os.path.join(os.getcwd(), 'results', 'rnn_lstm_model')
        os.makedirs(model_save_path, exist_ok=True)
        train_rnn_lstm_model(data_file, model_save_path)
    else:
        print('Invalid model choice. Use \"bert\" or \"rnn\".')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train BERT or RNN-LSTM model.')
    parser.add_argument('--model', type=str, required=True, help='Model to train: bert or rnn')
    args = parser.parse_args()
    main(args.model)
