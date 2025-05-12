import matplotlib.pyplot as plt
import numpy as np

# Categories and binary category indices
categories = ['Coherence', 'Engagingness', 'No Hallucinations', 'Completeness', 'Consistency',
              'Fluency', 'Correctness', 'Usefulness', 'Reasoning', 'Overall Satisfaction']
N = len(categories)
binary_indices = [2, 4]  # Hallucinations and Consistency

# Human scores
human_llama_1 = [2.50, 2.17, 1.00, 2.17, 1.00, 1.83, 3.33, 2.50, 2.50, 2.00]
human_openai_1 = [3.67, 3.00, 1.00, 3.17, 1.00, 4.17, 4.00, 3.00, 3.17, 3.17]
human_llama_2 = [2.42, 2.67, 1.00, 2.67, 0.83, 2.33, 3.67, 2.50, 2.83, 2.17]
human_openai_2 = [3.75, 3.17, 0.67, 3.17, 0.67, 4.33, 4.17, 3.33, 3.00, 2.83]

# LLM scores
llm_llama_1 = [3.00, 2.00, 1.00, 2.50, 1.00, 3.50, 4.00, 2.50, 3.00, 2.50]
llm_openai_1 = [4.50, 3.50, 1.00, 4.50, 1.00, 5.00, 5.00, 5.00, 4.50, 4.50]
llm_llama_2 = [2.50, 1.50, 1.00, 2.00, 1.00, 2.50, 3.50, 2.00, 2.00, 2.00]
llm_openai_2 = [4.00, 3.50, 1.00, 4.00, 1.00, 5.00, 5.00, 4.00, 4.00, 4.00]

# Adjust binary scores to 1–5 scale
def adjust_binary_to_scale_1_5(scores):
    return [1 + 4 * s if i in binary_indices and s <= 1 else s for i, s in enumerate(scores)]

# Apply normalization
all_scores = [
    human_llama_1, human_openai_1, human_llama_2, human_openai_2,
    llm_llama_1, llm_openai_1, llm_llama_2, llm_openai_2
]
all_scores = [adjust_binary_to_scale_1_5(s) for s in all_scores]
(
    human_llama_1, human_openai_1, human_llama_2, human_openai_2,
    llm_llama_1, llm_openai_1, llm_llama_2, llm_openai_2
) = all_scores

# Radar setup
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += [angles[0]]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw=dict(polar=True))
axes = axes.flatten()

chart_data = [
    ("LLaMa – Slide Paper 1", human_llama_1, llm_llama_1),
    ("OpenAI – Slide Paper 1", human_openai_1, llm_openai_1),
    ("LLaMa – Slide Paper 2", human_llama_2, llm_llama_2),
    ("OpenAI – Slide Paper 2", human_openai_2, llm_openai_2),
]

for ax, (title, human, llm) in zip(axes, chart_data):
    human += [human[0]]
    llm += [llm[0]]

    ax.plot(angles, human, label='Human', color='blue')
    ax.fill(angles, human, color='blue', alpha=0.1)

    ax.plot(angles, llm, label='LLM', color='orange')
    ax.fill(angles, llm, color='orange', alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([str(i) for i in range(1, 6)], fontsize=7)

    ax.set_title(title, size=11, y=1.05)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.1, 1.05))

plt.suptitle("Human vs LLM Evaluation – Slide Papers", fontsize=15, y=1.03)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
