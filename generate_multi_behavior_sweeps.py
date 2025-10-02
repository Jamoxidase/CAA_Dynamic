#!/usr/bin/env python3
"""Generate multi-behavior layer sweep plots for 7B models."""

import matplotlib.pyplot as plt
from plot_results import get_data, get_avg_key_prob
from steering_settings import SteeringSettings
from behaviors import ALL_BEHAVIORS
import os

def create_multi_behavior_sweep(layers, behaviors, use_base_model, model_size, output_file):
    """Create a single plot showing all behaviors across layers."""
    plt.clf()
    plt.figure(figsize=(8, 5))

    settings = SteeringSettings()
    settings.type = 'ab'
    settings.use_base_model = use_base_model
    settings.model_size = model_size

    for behavior in behaviors:
        if "coordinate" in behavior:
            continue
        settings.behavior = behavior
        pos_per_layer = []
        neg_per_layer = []

        for layer in sorted(layers):
            try:
                base_data = get_data(layer, 0, settings)
                pos_data = get_data(layer, 1, settings)
                neg_data = get_data(layer, -1, settings)

                if len(base_data) == 0 or len(pos_data) == 0 or len(neg_data) == 0:
                    print(f"[WARN] Missing data for {behavior} layer {layer}")
                    pos_per_layer.append(0)
                    neg_per_layer.append(0)
                    continue

                base_res = get_avg_key_prob(base_data, "answer_matching_behavior")
                pos_res = get_avg_key_prob(pos_data, "answer_matching_behavior") - base_res
                neg_res = get_avg_key_prob(neg_data, "answer_matching_behavior") - base_res
                pos_per_layer.append(pos_res)
                neg_per_layer.append(neg_res)
                print(f"{behavior} layer {layer}: pos_delta={pos_res:.3f}, neg_delta={neg_res:.3f}")
            except Exception as e:
                print(f"[ERROR] {behavior} layer {layer}: {e}")
                pos_per_layer.append(0)
                neg_per_layer.append(0)

        # Plot this behavior's lines
        plt.plot(
            sorted(layers),
            pos_per_layer,
            linestyle="solid",
            linewidth=2,
            color="#377eb8",
            alpha=0.7
        )
        plt.plot(
            sorted(layers),
            neg_per_layer,
            linestyle="solid",
            linewidth=2,
            color="#ff7f00",
            alpha=0.7
        )

    # Add legend
    plt.plot([], [], linestyle="solid", linewidth=2, color="#377eb8", label="Positive steering")
    plt.plot([], [], linestyle="solid", linewidth=2, color="#ff7f00", label="Negative steering")

    # Formatting
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel(r"$\Delta$ p(answer matching behavior)", fontsize=12)

    model_name = f"Llama 2 7B {'Base' if use_base_model else 'Chat'}"
    plt.title(f"Per-layer CAA effect: {model_name}", fontsize=14)

    # X-ticks: show all layers if <= 10, otherwise every 5
    if len(layers) <= 10:
        plt.xticks(ticks=sorted(layers), labels=sorted(layers))
    else:
        tick_layers = sorted(layers)[::5]
        if sorted(layers)[-1] not in tick_layers:
            tick_layers.append(sorted(layers)[-1])
        plt.xticks(ticks=tick_layers, labels=tick_layers)

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, format="png", dpi=150)
    print(f"\n[SUCCESS] Saved to {output_file}")

if __name__ == "__main__":
    behaviors = ["envy-kindness", "gluttony-temperance", "greed-charity",
                 "lust-chastity", "pride-humility", "sloth-diligence", "wrath-patience"]

    # 7B Base model
    print("=" * 60)
    print("Generating 7B Base multi-behavior layer sweep...")
    print("=" * 60)
    create_multi_behavior_sweep(
        layers=[10, 11, 12, 13],
        behaviors=behaviors,
        use_base_model=True,
        model_size="7b",
        output_file="analysis/MULTI_BEHAVIOR_LAYER_SWEEP_7b_base.png"
    )

    # 7B Chat model
    print("\n" + "=" * 60)
    print("Generating 7B Chat multi-behavior layer sweep...")
    print("=" * 60)
    create_multi_behavior_sweep(
        layers=list(range(0, 31)),
        behaviors=behaviors,
        use_base_model=False,
        model_size="7b",
        output_file="analysis/MULTI_BEHAVIOR_LAYER_SWEEP_7b_chat.png"
    )
