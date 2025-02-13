# src/visualize.py

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics():
    # Assumes metrics are stored in a CSV file in results/tables/metrics.csv
    metrics_file = os.path.join(os.getcwd(), 'results', 'tables', 'metrics.csv')
    if not os.path.exists(metrics_file):
        print('Metrics file not found!')
        return
    
    df = pd.read_csv(metrics_file)
    
    # Example: Plot accuracy for each dataset and model
    plt.figure(figsize=(10,6))
    for model in df['Model'].unique():
        subset = df[df['Model'] == model]
        plt.plot(subset['Dataset'], subset['Accuracy'], marker='o', label=model)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    fig_path = os.path.join(os.getcwd(), 'results', 'figures', 'accuracy_comparison.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.show()
    print(f'Figure saved to {fig_path}')

if __name__ == '__main__':
    plot_metrics()

