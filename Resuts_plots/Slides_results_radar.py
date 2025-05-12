import matplotlib.pyplot as plt
import numpy as np

# Categories and binary category indices
categories = ['Coherence', 'Engagingness', 'No Hallucinations', 'Completeness', 'Consistency',
              'Fluency', 'Correctness', 'Usefulness', 'Reasoning', 'Overall Satisfaction']
N = len(categories)
binary_indices = [2, 4]  # Hallucinations and Consistency

# Updated scores for Radar 1
llama_1 = [2.50, 2.17, 1.00, 2.17, 1.00, 1.83, 3.33, 2.50, 2.50, 2.00]
openai_1 = [3.67, 3.00, 1.00, 3.17, 1.00, 4.17, 4.00, 3.00, 3.17, 3.17]

# Updated scores for Radar 2
llama_2 = [2.42, 2.67, 1.00, 2.67, 0.83, 2.33, 3.67, 2.50, 2.83, 2.17]
openai_2 = [3.75, 3.17, 0.67, 3.17, 0.67, 4.33, 4.17, 3.33, 3.00, 2.83]


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
    ('Evaluation Slide Paper 1', llama_1, openai_1),
    ('Evaluation Slide Paper 2', llama_2, openai_2)
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
