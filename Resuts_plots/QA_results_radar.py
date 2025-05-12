import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

categories = ['Coherence', 'No Hallucinations', 'Relevance', 'Completeness', 'Consistency',
              'Fluency', 'Correctness', 'Usefulness', 'Reasoning', 'Overall Satisfaction']
N = len(categories)

# Raw scores
llama_graph = [4.06, 0.28, 3.11, 2.33, 0.61, 4.72, 2.94, 2.56, 3.22, 2.61]
openai_graph = [4.28, 0.94, 4.50, 4.78, 0.78, 4.44, 4.50, 4.06, 4.39, 4.06]
llama_rag = [4.56, 0.94, 3.33, 3.06, 1.00, 4.44, 4.67, 3.17, 3.33, 3.28]
openai_rag = [4.89, 0.94, 4.72, 4.78, 0.94, 4.72, 4.72, 4.56, 4.56, 4.50]


# Indices of binary categories (Hallucinations and Consistency)
binary_indices = [1, 4]

def adjust_binary_to_scale_1_5(scores):
    return [
        1 + 4 * s if i in binary_indices else s
        for i, s in enumerate(scores)
    ]

# Adjust scores
models = {
    'LLaMa Graph RAG': adjust_binary_to_scale_1_5(llama_graph),
    'OpenAI Graph RAG': adjust_binary_to_scale_1_5(openai_graph),
    'LLaMa RAG': adjust_binary_to_scale_1_5(llama_rag),
    'OpenAI RAG': adjust_binary_to_scale_1_5(openai_rag),
}

# Radar setup
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += [angles[0]]

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

for model_name, score_list in models.items():
    values = score_list + [score_list[0]]
    ax.plot(angles, values, label=model_name)
    ax.fill(angles, values, alpha=0.1)

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)

# Set range to 1–5
ax.set_ylim(1, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels([str(i) for i in range(1, 6)])

# Title and legend
plt.title('Models Comparison Across 10 Evaluation Categories (1–5 Scale)', size=14, y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert the models dictionary to a DataFrame
df_heatmap = pd.DataFrame(models, index=categories).T

# Plot the heatmap
plt.figure(figsize=(12, 5))
sns.heatmap(df_heatmap, annot=True, cmap='YlGnBu', linewidths=0.5, linecolor='gray', vmin=1, vmax=5)
plt.title('Heatmap of Evaluation Scores (1–5 Scale)', fontsize=14)
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

