from main import Logger
import numpy as np
import matplotlib.pyplot as plt

config = {'metric': 'L1',
          'val_sample': 5000,
          'train_sample': 50000,
          'test_sample': 10000}

def get_data(data_type: str, config: dict):
    match data_type:
        case 'Test':
            logger = Logger('test_accuracy.txt', data_type, config=config)
            sample = config['test_sample']
        case 'Val': 
            logger = Logger('val_accuracy.txt', data_type, config=config)
            sample = config['val_sample']
        case _: 
            logger = Logger('test_accuracy.txt', 'Test', config=config)
            sample = config['test_sample']

    k, accuracy = [], []
    for key, acc in logger.record.items():
        if key[0] == sample:
            k.append(key[1])
            accuracy.append(acc)
    
    return k, accuracy

def test_plot():
    configs = [
        {'metric': 'L1', 'val_sample': 5000, 'train_sample': 50000, 
         'test_sample': 10000, 'label': 'L1 Distance'},
        {'metric': 'L2', 'val_sample': 5000, 'train_sample': 50000, 
         'test_sample': 10000, 'label': 'L2 Distance'}]
    
    plt.figure(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72']
    for i, config in enumerate(configs):
        Test_K, Test_Accuracy = get_data('Test', config)
        
        plt.xticks(Test_K)
        plt.ylim(0.30, 0.40)
        plt.plot(Test_K, Test_Accuracy, 
                marker='o', color=colors[i], 
                linestyle='-', label=config['label'],
                linewidth=1.0, markersize=4,
                markerfacecolor='white', markeredgewidth=0.8,
                alpha=0.9)
    
    plt.xlabel('K Value', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Test Performance With Different Parameters', fontsize=13, pad=15)
    plt.legend(frameon=True, framealpha=0.95, fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.2, linewidth=0.5)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    
    plt.tight_layout()
    plt.show()

def val_plot():
    val_sizes = [1000, 2000, 5000, 10000]
    colors = ['#FF6B6B', '#4ECDC4', '#556270', "#37B166"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, val_size in enumerate(val_sizes):
        config = {'metric': 'L1', 'val_sample': val_size, 
                 'train_sample': 50000, 'test_sample': 10000}
        Val_K, Val_Accuracy = get_data('Val', config)
        
        ax1.set_ylim(0.30, 0.40)
        ax1.plot(Val_K, Val_Accuracy, 
                marker='o', color=colors[i], 
                linestyle='-', linewidth=1.0,
                markersize=5, label=f'Val{val_size}')
    
    for i, val_size in enumerate(val_sizes):
        config = {'metric': 'L2', 'val_sample': val_size, 
                 'train_sample': 50000, 'test_sample': 10000}
        Val_K, Val_Accuracy = get_data('Val', config)
        
        ax2.set_ylim(0.30, 0.40) 
        ax2.plot(Val_K, Val_Accuracy, 
                marker='s', color=colors[i], 
                linestyle='-', linewidth=1.0,
                markersize=5, label=f'Val{val_size}')
    
    for ax, title in zip([ax1, ax2], ['L1 Distance', 'L2 Distance']):
        ax.set_xlabel('K Value')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title(title)
        ax.legend(loc = 'lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def val_errorbar(metric):
    val_sizes = [1000, 2000, 5000, 7000, 10000]
    configs = [{'metric': metric, 'val_sample': val_size, 
                'train_sample': 50000, 'test_sample': 10000} 
                for val_size in val_sizes]
    
    all_data = {f'Val_{config["val_sample"]}': {'k': [], 'acc': []} for config in configs}
    k_accuracies = {k: [] for k in range(1, 21)}
    
    for config in configs:
        Val_K, Val_Accuracy = get_data('Val', config)
        label = f'Val_{config["val_sample"]}'
        for k, acc in zip(Val_K, Val_Accuracy):
            k_accuracies[k].append(acc)
            all_data[label]['k'].append(k)
            all_data[label]['acc'].append(acc)
    
    k_values = [k for k in range(1, 21) if k_accuracies[k]]
    mean_acc = [np.mean(k_accuracies[k]) for k in k_values]
    std_acc = [np.std(k_accuracies[k]) for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.xticks(k_values)
    plt.ylim(0.29, 0.37)
    plt.errorbar(k_values, mean_acc, yerr=std_acc,
                 fmt='o-', color='#2E86AB', linewidth=1.2,
                 markersize=5, capsize=3, alpha=0.8,
                 label='Mean ± Std', zorder=2)
    
    colors = ["#F14040", "#D769E1", '#556270', "#4ABA66", "#D98714"]
    for i, (label, color) in enumerate(zip(all_data.keys(), colors)):
        plt.scatter(all_data[label]['k'], all_data[label]['acc'],
                   color=color, marker='o', s=30, alpha=0.8,
                   label=label, zorder=3)
    
    plt.xlabel('K Value')
    plt.ylabel('Validation Accuracy (%)')
    plt.title(f'Validation Performance With Different Size In {metric} Distance')
    plt.legend(loc = 'lower right')
    plt.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    val_errorbar('L2')