# CAA Analysis Report: LFM2 1.2B Model

**Model**: LiquidAI LFM-2 1.2B
**Behaviors**: Seven Deadly Sins (envy, gluttony, greed, lust, pride, sloth, wrath)
**Layers Analyzed**: 0-5 (6 layers total)
**Multipliers**: -5 to +5 (AB tests), -5 to +5 (open-ended)
**Date**: October 1, 2025

---

## 1. Overview: Multi-Behavior Layer Sweep

Change in p(answer matching behavior) from baseline (multiplier=0) using multipliers ±1 across all layers for all seven behaviors.

![Multi-behavior layer sweep](LAYER_SWEEPS_behavior=envy-kindness_type=ab_use_base_model=False_model_size=1.2b.png)

**Key observations**:
- Negative steering (orange) shows stronger effects than positive steering (blue)
- Peak effects occur around layers 2-3 for most behaviors
- Maximum Δp ~30% for negative steering at layer 3
- Effects generally diminish at layer 5

---

## 2. Per-Behavior AB Test Results

### 2.1 Envy-Kindness

#### AB Layer Sweep (multipliers ±1)
![Envy-Kindness AB](envy-kindness/behavior=envy-kindness_type=ab_use_base_model=False_model_size=1.2b.png)

**Steering effects**:
- Positive steering (+1): p(kindness) increases from ~54% to ~58% (layer 5 peak)
- Negative steering (-1): p(envy) peaks at 63% (layer 0), drops to 46% (layer 1), recovers to 61% (layer 5)
- Layer 1 shows strongest suppression effect

#### Open-Ended Behavioral Scores by Layer

![Layer 0](envy-kindness/layer=0_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 0**: Scores range from 2.0/10 (envious) to 5.5/10 (neutral-kind). Moderate steering control.

![Layer 1](envy-kindness/layer=1_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 1**: Scores range from 1.5/10 to 5.5/10. Similar range to layer 0.

![Layer 2](envy-kindness/layer=2_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 2**: Scores range from 1.5/10 to 6.0/10. Slightly improved positive steering.

![Layer 3](envy-kindness/layer=3_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 3**: Scores range from 0.8/10 to 6.5/10. **Best layer for steering control** - widest range.

![Layer 4](envy-kindness/layer=4_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 4**: Scores range from 1.0/10 to 6.5/10. Strong steering similar to layer 3.

![Layer 5](envy-kindness/layer=5_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 5**: Scores range from 1.5/10 to 6.5/10. Good control maintained.

---

### 2.2 Gluttony-Temperance

#### AB Layer Sweep (multipliers ±1)
![Gluttony-Temperance AB](gluttony-temperance/behavior=gluttony-temperance_type=ab_use_base_model=False_model_size=1.2b.png)

**Steering effects**:
- Positive steering (+1): p(temperance) decreases from 53% to 49% (layer 3 minimum)
- Negative steering (-1): p(gluttony) increases from 55% to 74% (layer 3 peak)
- **Strong negative steering effect** - one of the most steerable behaviors

#### Open-Ended Behavioral Scores by Layer

![Layer 0](gluttony-temperance/layer=0_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 0**: Scores range from 2.0/10 to 5.0/10. Moderate control.

![Layer 1](gluttony-temperance/layer=1_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 1**: Scores range from 1.5/10 to 5.0/10. Similar to layer 0.

![Layer 2](gluttony-temperance/layer=2_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 2**: Scores range from 1.5/10 to 5.5/10. Improved positive steering.

![Layer 3](gluttony-temperance/layer=3_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 3**: Scores range from 1.0/10 to 5.5/10. Good control range.

![Layer 4](gluttony-temperance/layer=4_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 4**: Scores range from 1.0/10 to 5.5/10. Consistent with layer 3.

![Layer 5](gluttony-temperance/layer=5_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 5**: Scores range from 1.0/10 to 5.5/10. Control maintained.

---

### 2.3 Greed-Charity

#### AB Layer Sweep (multipliers ±1)
![Greed-Charity AB](greed-charity/behavior=greed-charity_type=ab_use_base_model=False_model_size=1.2b.png)

**Steering effects**:
- Positive steering (+1): p(charity) relatively flat around 50-52%
- Negative steering (-1): p(greed) increases from 53% to 64% (layer 3 peak)
- Moderate steering effectiveness

#### Open-Ended Behavioral Scores by Layer

![Layer 0](greed-charity/layer=0_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 0**: Scores range from 2.5/10 to 5.0/10. Limited steering range.

![Layer 1](greed-charity/layer=1_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 1**: Scores range from 2.0/10 to 5.0/10. Similar to layer 0.

![Layer 2](greed-charity/layer=2_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 2**: Scores range from 2.0/10 to 5.5/10. Slight improvement.

![Layer 3](greed-charity/layer=3_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 3**: Scores range from 1.5/10 to 5.5/10. Best control for this behavior.

![Layer 4](greed-charity/layer=4_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 4**: Scores range from 1.5/10 to 5.5/10. Consistent with layer 3.

![Layer 5](greed-charity/layer=5_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 5**: Scores range from 1.5/10 to 5.5/10. Control maintained.

---

### 2.4 Lust-Chastity

#### AB Layer Sweep (multipliers ±1)
![Lust-Chastity AB](lust-chastity/behavior=lust-chastity_type=ab_use_base_model=False_model_size=1.2b.png)

**Steering effects**:
- Positive steering (+1): p(chastity) relatively stable around 50-52%
- Negative steering (-1): p(lust) increases from 51% to 60% (layer 1 peak)
- Moderate steering with early-layer peak

#### Open-Ended Behavioral Scores by Layer

![Layer 0](lust-chastity/layer=0_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 0**: Scores range from 2.0/10 to 5.5/10. Moderate control.

![Layer 1](lust-chastity/layer=1_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 1**: Scores range from 1.5/10 to 5.5/10. Similar range.

![Layer 2](lust-chastity/layer=2_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 2**: Scores range from 1.5/10 to 5.5/10. Consistent control.

![Layer 3](lust-chastity/layer=3_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 3**: Scores range from 1.0/10 to 6.0/10. Good steering range.

![Layer 4](lust-chastity/layer=4_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 4**: Scores range from 1.0/10 to 6.0/10. Similar to layer 3.

![Layer 5](lust-chastity/layer=5_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 5**: Scores range from 1.0/10 to 6.0/10. Control maintained.

---

### 2.5 Pride-Humility

#### AB Layer Sweep (multipliers ±1)
![Pride-Humility AB](pride-humility/behavior=pride-humility_type=ab_use_base_model=False_model_size=1.2b.png)

**Steering effects**:
- Positive steering (+1): p(humility) decreases from 56% to 50% (layer 3)
- Negative steering (-1): p(pride) increases from 50% to 67% (layer 4 peak)
- Good negative steering control

#### Open-Ended Behavioral Scores by Layer

![Layer 0](pride-humility/layer=0_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 0**: Scores range from 2.0/10 to 5.5/10. Moderate control.

![Layer 1](pride-humility/layer=1_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 1**: Scores range from 1.5/10 to 5.5/10. Similar range.

![Layer 2](pride-humility/layer=2_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 2**: Scores range from 1.5/10 to 5.5/10. Consistent.

![Layer 3](pride-humility/layer=3_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 3**: Scores range from 1.0/10 to 6.0/10. Good control range.

![Layer 4](pride-humility/layer=4_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 4**: Scores range from 1.0/10 to 6.0/10. Similar to layer 3.

![Layer 5](pride-humility/layer=5_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 5**: Scores range from 1.0/10 to 6.0/10. Control maintained.

---

### 2.6 Sloth-Diligence

#### AB Layer Sweep (multipliers ±1)
![Sloth-Diligence AB](sloth-diligence/behavior=sloth-diligence_type=ab_use_base_model=False_model_size=1.2b.png)

**Steering effects**:
- Positive steering (+1): p(diligence) relatively flat around 51-54%
- Negative steering (-1): p(sloth) increases from 50% to 63% (layer 2 peak)
- Moderate steering effectiveness

#### Open-Ended Behavioral Scores by Layer

![Layer 0](sloth-diligence/layer=0_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 0**: Scores range from 2.5/10 to 5.5/10. Moderate control.

![Layer 1](sloth-diligence/layer=1_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 1**: Scores range from 2.0/10 to 5.5/10. Similar range.

![Layer 2](sloth-diligence/layer=2_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 2**: Scores range from 1.5/10 to 5.5/10. Improved negative steering.

![Layer 3](sloth-diligence/layer=3_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 3**: Scores range from 1.5/10 to 6.0/10. Good control range.

![Layer 4](sloth-diligence/layer=4_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 4**: Scores range from 1.5/10 to 6.0/10. Consistent with layer 3.

![Layer 5](sloth-diligence/layer=5_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 5**: Scores range from 1.5/10 to 6.0/10. Control maintained.

---

### 2.7 Wrath-Patience

#### AB Layer Sweep (multipliers ±1)
![Wrath-Patience AB](wrath-patience/behavior=wrath-patience_type=ab_use_base_model=False_model_size=1.2b.png)

**Steering effects**:
- Positive steering (+1): p(patience) relatively stable around 63-66%
- Negative steering (-1): p(wrath) increases dramatically from 48% to 70% (layer 4 peak)
- **Strongest negative steering effect** among all behaviors

#### Open-Ended Behavioral Scores by Layer

![Layer 0](wrath-patience/layer=0_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 0**: Scores range from 2.5/10 to 5.5/10. Moderate control.

![Layer 1](wrath-patience/layer=1_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 1**: Scores range from 2.0/10 to 5.5/10. Similar range.

![Layer 2](wrath-patience/layer=2_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 2**: Scores range from 1.5/10 to 6.0/10. Improved control.

![Layer 3](wrath-patience/layer=3_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 3**: Scores range from 1.0/10 to 6.5/10. Excellent control range.

![Layer 4](wrath-patience/layer=4_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 4**: Scores range from 1.0/10 to 6.5/10. **Best layer** for this behavior.

![Layer 5](wrath-patience/layer=5_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=1.2b.png)
**Layer 5**: Scores range from 1.0/10 to 6.5/10. Strong control maintained.

---

## 3. Summary of Findings

### 3.1 Best Layers for Steering
- **Layers 3-4**: Most effective for steering control across all behaviors
- **Layer 1**: Special case for envy-kindness (strongest suppression)
- **Layer 5**: Effects diminish but remain present

### 3.2 Steering Effectiveness by Behavior
**Strongest negative steering** (easiest to induce vice):
1. Wrath (Δp up to +22% at layer 4)
2. Gluttony (Δp up to +19% at layer 3)
3. Pride (Δp up to +17% at layer 4)

**Weakest negative steering**:
1. Lust (Δp up to +9% at layer 1)
2. Greed (Δp up to +11% at layer 3)

**Positive steering** (inducing virtue):
- Generally weaker than negative steering
- Most behaviors show only 3-5% improvement
- Positive steering effectiveness relatively uniform across behaviors

### 3.3 Open-Ended Score Ranges
- **Best control**: Envy-kindness, wrath-patience (0.8-6.5/10 range)
- **Moderate control**: Most other behaviors (1.0-6.0/10 range)
- **Limited high scores**: Difficulty steering toward strong virtuous behavior (max ~6.5/10)
- **Strong vice induction**: Can reliably achieve scores <1.5/10

### 3.4 Model Size Observations
The 1.2B model shows:
- Clear steering effects despite small size
- Stronger negative (vice) steering than positive (virtue) steering
- Layer-dependent effectiveness with mid-layers (2-4) most effective
- Limited ability to achieve highly virtuous outputs (capped around 6.5/10)

---

## 4. Conclusions

The LFM2 1.2B model demonstrates clear CAA steering effectiveness across all seven deadly sin behaviors:

1. **Negative steering is more effective** than positive steering across all behaviors
2. **Mid-layers (2-4) are optimal** for most steering interventions
3. **Wrath and gluttony** are the most steerable behaviors
4. **Vice induction is easier** than virtue induction for this model
5. Despite small model size, **steering effects are substantial** (up to 22% Δp)

The model's steering characteristics suggest that the behavioral representations are well-formed even at 1.2B parameters, though the asymmetry between vice and virtue steering indicates potential training distribution biases or representational capacity limits for strongly virtuous behavior.
