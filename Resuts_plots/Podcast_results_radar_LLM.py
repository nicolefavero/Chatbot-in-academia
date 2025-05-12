import matplotlib.pyplot as plt
import numpy as np

# Categories and binary category indices
categories = ['Coherence', 'Engagingness', 'No Hallucinations', 'Completeness', 'Consistency',
              'Fluency', 'Correctness', 'Usefulness', 'Reasoning', 'Overall Satisfaction']
N = len(categories)
binary_indices = [2, 4]  # Hallucinations and Consistency

# Updated scores for Radar 1
llama_1 = [4.00, 3.50, 1.00, 3.50, 1.00, 5.00, 4.50, 3.50, 3.50, 3.50]
openai_1 = [4.50, 4.50, 1.00, 3.50, 1.00, 5.00, 4.50, 4.00, 4.00, 4.00]

# Updated scores for Radar 2
llama_2 = [4.00, 3.50, 1.00, 3.50, 1.00, 5.00, 4.50, 4.00, 4.00, 4.00]
openai_2 = [3.75, 4.50, 1.00, 3.00, 1.00, 5.00, 4.00, 3.50, 3.50, 3.50]

# Adjust binary scores from 0–1 to 1–5 scale
def adjust_binary_to_scale_1_5(scores):
    return [
        1 + 4 * s if i in binary_indices and s <= 1 else s
        for i, s in enumerate(scores)
    ]

# Normalize all values
llama_1 = adjust_binary_to_scale_1_5(llama_1)
openai_1 = adjust_binary_to_scale_1_5(openai_1)
llama_2 = adjust_binary_to_scale_1_5(llama_2)
openai_2 = adjust_binary_to_scale_1_5(openai_2)

# Radar chart setup
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += [angles[0]]

fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(polar=True))

# Data for both charts
data_pairs = [
    ('Evaluation Podcast Paper 1', llama_1, openai_1),
    ('Evaluation Podcast Paper 2', llama_2, openai_2)
]

for ax, (title, llama, openai) in zip(axes, data_pairs):
    llama += [llama[0]]
    openai += [openai[0]]

    ax.plot(angles, llama, label='LLaMa', color='blue')
    ax.fill(angles, llama, color='blue', alpha=0.1)

    ax.plot(angles, openai, label='OpenAI', color='green')
    ax.fill(angles, openai, color='green', alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([str(i) for i in range(1, 6)], fontsize=8)
    ax.set_title(title, size=14, y=1.1)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.tight_layout()
plt.show()
