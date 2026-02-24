import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def do_topsis(df, w, imp):
    models = df.iloc[:, 0]
    data = df.iloc[:, 1:].astype(float)

    # 1. Normalize
    norms = np.linalg.norm(data, axis=0)
    norm_data = data / norms

    # 2. Mul by weights
    weighted = norm_data * w

    # 3. Find the ideal best and worst values --> no loops
    is_pos = np.array(imp) == '+'
    
    best_ideal = np.where(is_pos, weighted.max(axis=0), weighted.min(axis=0))
    worst_ideal = np.where(is_pos, weighted.min(axis=0), weighted.max(axis=0))

    # 4. Calc dist from the ideals
    dist_best = np.linalg.norm(weighted.values - best_ideal, axis=1)
    dist_worst = np.linalg.norm(weighted.values - worst_ideal, axis=1)

    # 5. get the topsis score
    score = dist_worst / (dist_best + dist_worst)

    # 6. rank it
    res = df.copy()
    res['Score'] = score
    res['Rank'] = res['Score'].rank(ascending=False).astype(int)
    
    return res.sort_values('Rank')

def make_chart(df):
    plt.figure(figsize=(9, 5))
    
    # Sort it so the best is at the top of the chart
    chart_data = df.sort_values('Score', ascending=True)
    
    plt.barh(chart_data.iloc[:, 0], chart_data['Score'], color='cornflowerblue')
    plt.xlabel('TOPSIS Score')
    plt.title('Best LLMs for Text Generation')
    
    # Add num to the bars
    for i, val in enumerate(chart_data['Score']):
        plt.text(val, i, f" {round(val, 3)}")
        
    plt.tight_layout()
    plt.savefig('my_topsis_chart.png')
    plt.show()

if __name__ == "__main__":
    #text gen model selection data
    raw_data = {
        'Model': ['Llama-3-8B', 'Mistral-7B', 'Gemma-7B', 'Falcon-7B', 'GPT-2-1.5B'],
        'Params_B': [8.0, 7.3, 8.5, 7.0, 1.5],
        'Tokens': [8192, 8192, 8192, 2048, 1024],
        'MMLU': [68.4, 62.5, 64.3, 47.9, 25.0],
        'HumanEval': [62.2, 30.5, 32.3, 15.0, 0.0],
        'Latency': [25, 28, 30, 22, 10]
    }
    
    df = pd.DataFrame(raw_data)
    df.to_csv('base_data.csv', index=False) 

    # weight and impact 
    weights = [1, 1, 2, 2, 1]
    impacts = ['-', '+', '+', '+', '-']

    ans = do_topsis(df, weights, impacts)
    ans.to_csv('final_results.csv', index=False)
    
    print("Results:")
    print(ans)
    
    make_chart(ans)