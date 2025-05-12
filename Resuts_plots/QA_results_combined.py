import matplotlib.pyplot as plt
import numpy as np

# Categories
categories = ['Coherence', 'No Hallucinations', 'Relevance', 'Completeness', 'Consistency',
              'Fluency', 'Correctness', 'Usefulness', 'Reasoning', 'Overall Satisfaction']
N = len(categories)

# Indices of binary categories
binary_indices = [1, 4]

def adjust_binary_to_scale_1_5(scores):
    return [1 + 4 * s if i in binary_indices else s for i, s in enumerate(scores)]

# Raw human scores
human_scores = {
    'LLaMa Graph': [4.06, 0.28, 3.11, 2.33, 0.61, 4.72, 2.94, 2.56, 3.22, 2.61],
    'OpenAI Graph': [4.28, 0.94, 4.50, 4.78, 0.78, 4.44, 4.50, 4.06, 4.39, 4.06],
    'LLaMa RAG': [4.56, 0.94, 3.33, 3.06, 1.00, 4.44, 4.67, 3.17, 3.33, 3.28],
    'OpenAI RAG': [4.89, 0.94, 4.72, 4.78, 0.94, 4.72, 4.72, 4.56, 4.56, 4.50]
}

# Raw LLM scores
llm_scores = {
    'LLaMa Graph': [4.33, 0.00, 2.00, 1.33, 0.00, 4.67, 1.33, 1.50, 1.83, 1.50],
    'OpenAI Graph': [4.67, 0.83, 5.00, 5.00, 1.00, 4.83, 4.83, 5.00, 5.00, 4.67],
    'LLaMa RAG': [4.50, 1.00, 3.83, 3.17, 1.00, 4.50, 4.33, 3.17, 3.50, 3.33],
    'OpenAI RAG': [5.00, 1.00, 5.00, 4.83, 1.00, 5.00, 5.00, 4.83, 4.67, 4.83]
}

# Adjust binary scores
for model in human_scores:
    human_scores[model] = adjust_binary_to_scale_1_5(human_scores[model])
    llm_scores[model] = adjust_binary_to_scale_1_5(llm_scores[model])

# Radar setup
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += [angles[0]]

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(13, 10), subplot_kw=dict(polar=True))
axs = axs.flatten()

for idx, model in enumerate(human_scores):
    ax = axs[idx]
    human = human_scores[model] + [human_scores[model][0]]
    llm = llm_scores[model] + [llm_scores[model][0]]

    ax.plot(angles, human, label='Human', color='blue')
    ax.fill(angles, human, color='blue', alpha=0.1)

    ax.plot(angles, llm, label='LLM', color='orange')
    ax.fill(angles, llm, color='orange', alpha=0.1)

    ax.set_title(model, size=11, y=1.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([str(i) for i in range(1, 6)], fontsize=7)

    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.1, 1.05))

# Reduce overlap
plt.suptitle('Human vs LLM Scores per Model (1–5 Scale)', fontsize=15, y=1.03)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
