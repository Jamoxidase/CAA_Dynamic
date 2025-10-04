# CAA Analysis Report: Llama 2 7B Chat Model

**Model**: Llama 2 7B Chat
**Behaviors**: Seven Deadly Sins (envy, gluttony, greed, lust, pride, sloth, wrath)
**Layers Analyzed**: 0-30 (31 layers total)
**Multipliers**: -5 to +5 (AB tests)
**Date**: October 1, 2025

---

## 1. Overview: Multi-Behavior Layer Sweep

Change in p(answer matching behavior) from baseline (multiplier=0) using multipliers ±1 across all layers for all seven behaviors.

<p align="center">
  <img src="LAYER_SWEEPS_behavior=envy-kindness_type=ab_use_base_model=False_model_size=7b.png" width="800" alt="Multi-behavior layer sweep">
</p>

**Key observations**:
- **Layer 16-17 boundary**: Dramatic positive steering collapse - all blue lines drop to ~0%
- **Layers 0-15**: Strong positive steering (0-60%), highly variable negative steering (20-90%)
- **Layers 17-30**: Positive steering eliminated (<5%), negative steering persists (10-30%)
- **Peak effects**: Layers 9-15 show highest positive steering (40-60% delta)

This dramatic asymmetry demonstrates that chat fine-tuning created layer-localized virtue representations that become inaccessible after layer 16.

---

## 2. Per-Behavior Layer Sweeps (Absolute Probability)

These plots show p(answer matching behavior) with multipliers +1/-1 across all 31 layers.

<table>
<tr>
<td width="50%">

### Envy-Kindness
<img src="envy-kindness/behavior=envy-kindness_type=ab_use_base_model=False_model_size=7b.png" width="100%">

**Observations**: Positive peaks at layer 13 (62%), collapses after layer 17

</td>
<td width="50%">

### Gluttony-Temperance
<img src="gluttony-temperance/behavior=gluttony-temperance_type=ab_use_base_model=False_model_size=7b.png" width="100%">

**Observations**: Positive peaks at layer 9 (58%), similar collapse pattern

</td>
</tr>
<tr>
<td width="50%">

### Greed-Charity
<img src="greed-charity/behavior=greed-charity_type=ab_use_base_model=False_model_size=7b.png" width="100%">

**Observations**: Strong early-layer effect, peaks at layer 13 (64%)

</td>
<td width="50%">

### Lust-Chastity
<img src="lust-chastity/behavior=lust-chastity_type=ab_use_base_model=False_model_size=7b.png" width="100%">

**Observations**: Best at layers 0-1 (58-60%), early-layer advantage

</td>
</tr>
<tr>
<td width="50%">

### Pride-Humility
<img src="pride-humility/behavior=pride-humility_type=ab_use_base_model=False_model_size=7b.png" width="100%">

**Observations**: Excellent layers 0-15 (48-65%), most layer-sensitive

</td>
<td width="50%">

### Sloth-Diligence
<img src="sloth-diligence/behavior=sloth-diligence_type=ab_use_base_model=False_model_size=7b.png" width="100%">

**Observations**: Bimodal pattern - strong at 0-1 and 9-16

</td>
</tr>
<tr>
<td colspan="2" align="center">

### Wrath-Patience
<img src="wrath-patience/behavior=wrath-patience_type=ab_use_base_model=False_model_size=7b.png" width="50%">

**Observations**: Strongest early layers (0-2: 58-65%), excellent mid-layers (9-16)

</td>
</tr>
</table>

---

## 3. Multiplier Response at Layer 15

Representative layer showing steering response across different multiplier strengths.

<table>
<tr>
<td width="33%"><img src="layer=15_behavior=envy-kindness_type=ab_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="layer=15_behavior=gluttony-temperance_type=ab_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="layer=15_behavior=greed-charity_type=ab_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Envy-Kindness</b></small></td>
<td align="center"><small><b>Gluttony-Temperance</b></small></td>
<td align="center"><small><b>Greed-Charity</b></small></td>
</tr>
<tr>
<td width="33%"><img src="layer=15_behavior=lust-chastity_type=ab_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="layer=15_behavior=pride-humility_type=ab_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="layer=15_behavior=sloth-diligence_type=ab_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Lust-Chastity</b></small></td>
<td align="center"><small><b>Pride-Humility</b></small></td>
<td align="center"><small><b>Sloth-Diligence</b></small></td>
</tr>
<tr>
<td colspan="3" align="center"><img src="layer=15_behavior=wrath-patience_type=ab_use_base_model=False_model_size=7b.png" width="33%"></td>
</tr>
<tr>
<td colspan="3" align="center"><small><b>Wrath-Patience</b></small></td>
</tr>
</table>

---

## 4. Open-Ended Behavioral Scores by Layer

LLM judge scores (0-10 scale, Haiku judge) across multiplier strengths for open-ended generation tasks.

### Layer 9
<table>
<tr>
<td width="33%"><img src="envy-kindness/layer=9_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="gluttony-temperance/layer=9_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="greed-charity/layer=9_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Envy-Kindness</b></small></td>
<td align="center"><small><b>Gluttony-Temperance</b></small></td>
<td align="center"><small><b>Greed-Charity</b></small></td>
</tr>
<tr>
<td width="33%"><img src="lust-chastity/layer=9_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="pride-humility/layer=9_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="sloth-diligence/layer=9_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Lust-Chastity</b></small></td>
<td align="center"><small><b>Pride-Humility</b></small></td>
<td align="center"><small><b>Sloth-Diligence</b></small></td>
</tr>
<tr>
<td colspan="3" align="center"><img src="wrath-patience/layer=9_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=7b.png" width="33%"></td>
</tr>
<tr>
<td colspan="3" align="center"><small><b>Wrath-Patience</b></small></td>
</tr>
</table>

### Layer 10
<table>
<tr>
<td width="33%"><img src="envy-kindness/layer=10_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="gluttony-temperance/layer=10_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="greed-charity/layer=10_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Envy-Kindness</b></small></td>
<td align="center"><small><b>Gluttony-Temperance</b></small></td>
<td align="center"><small><b>Greed-Charity</b></small></td>
</tr>
<tr>
<td width="33%"><img src="lust-chastity/layer=10_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="pride-humility/layer=10_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="sloth-diligence/layer=10_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Lust-Chastity</b></small></td>
<td align="center"><small><b>Pride-Humility</b></small></td>
<td align="center"><small><b>Sloth-Diligence</b></small></td>
</tr>
<tr>
<td colspan="3" align="center"><img src="wrath-patience/layer=10_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=7b.png" width="33%"></td>
</tr>
<tr>
<td colspan="3" align="center"><small><b>Wrath-Patience</b></small></td>
</tr>
</table>

### Layer 11
<table>
<tr>
<td width="33%"><img src="envy-kindness/layer=11_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="gluttony-temperance/layer=11_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="greed-charity/layer=11_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Envy-Kindness</b></small></td>
<td align="center"><small><b>Gluttony-Temperance</b></small></td>
<td align="center"><small><b>Greed-Charity</b></small></td>
</tr>
<tr>
<td width="33%"><img src="lust-chastity/layer=11_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="pride-humility/layer=11_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="sloth-diligence/layer=11_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Lust-Chastity</b></small></td>
<td align="center"><small><b>Pride-Humility</b></small></td>
<td align="center"><small><b>Sloth-Diligence</b></small></td>
</tr>
<tr>
<td colspan="3" align="center"><img src="wrath-patience/layer=11_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=7b.png" width="33%"></td>
</tr>
<tr>
<td colspan="3" align="center"><small><b>Wrath-Patience</b></small></td>
</tr>
</table>

### Layer 12
<table>
<tr>
<td width="33%"><img src="envy-kindness/layer=12_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="gluttony-temperance/layer=12_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="greed-charity/layer=12_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Envy-Kindness</b></small></td>
<td align="center"><small><b>Gluttony-Temperance</b></small></td>
<td align="center"><small><b>Greed-Charity</b></small></td>
</tr>
<tr>
<td width="33%"><img src="lust-chastity/layer=12_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="pride-humility/layer=12_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="sloth-diligence/layer=12_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Lust-Chastity</b></small></td>
<td align="center"><small><b>Pride-Humility</b></small></td>
<td align="center"><small><b>Sloth-Diligence</b></small></td>
</tr>
<tr>
<td colspan="3" align="center"><img src="wrath-patience/layer=12_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=7b.png" width="33%"></td>
</tr>
<tr>
<td colspan="3" align="center"><small><b>Wrath-Patience</b></small></td>
</tr>
</table>

### Layer 13
<table>
<tr>
<td width="33%"><img src="envy-kindness/layer=13_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="gluttony-temperance/layer=13_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="greed-charity/layer=13_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Envy-Kindness</b></small></td>
<td align="center"><small><b>Gluttony-Temperance</b></small></td>
<td align="center"><small><b>Greed-Charity</b></small></td>
</tr>
<tr>
<td width="33%"><img src="lust-chastity/layer=13_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="pride-humility/layer=13_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="sloth-diligence/layer=13_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Lust-Chastity</b></small></td>
<td align="center"><small><b>Pride-Humility</b></small></td>
<td align="center"><small><b>Sloth-Diligence</b></small></td>
</tr>
<tr>
<td colspan="3" align="center"><img src="wrath-patience/layer=13_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=7b.png" width="33%"></td>
</tr>
<tr>
<td colspan="3" align="center"><small><b>Wrath-Patience</b></small></td>
</tr>
</table>

### Layer 14
<table>
<tr>
<td width="33%"><img src="envy-kindness/layer=14_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="gluttony-temperance/layer=14_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="greed-charity/layer=14_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Envy-Kindness</b></small></td>
<td align="center"><small><b>Gluttony-Temperance</b></small></td>
<td align="center"><small><b>Greed-Charity</b></small></td>
</tr>
<tr>
<td width="33%"><img src="lust-chastity/layer=14_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="pride-humility/layer=14_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="sloth-diligence/layer=14_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Lust-Chastity</b></small></td>
<td align="center"><small><b>Pride-Humility</b></small></td>
<td align="center"><small><b>Sloth-Diligence</b></small></td>
</tr>
<tr>
<td colspan="3" align="center"><img src="wrath-patience/layer=14_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=7b.png" width="33%"></td>
</tr>
<tr>
<td colspan="3" align="center"><small><b>Wrath-Patience</b></small></td>
</tr>
</table>

### Layer 15
<table>
<tr>
<td width="33%"><img src="envy-kindness/layer=15_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="gluttony-temperance/layer=15_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="greed-charity/layer=15_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Envy-Kindness</b></small></td>
<td align="center"><small><b>Gluttony-Temperance</b></small></td>
<td align="center"><small><b>Greed-Charity</b></small></td>
</tr>
<tr>
<td width="33%"><img src="lust-chastity/layer=15_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="pride-humility/layer=15_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="sloth-diligence/layer=15_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Lust-Chastity</b></small></td>
<td align="center"><small><b>Pride-Humility</b></small></td>
<td align="center"><small><b>Sloth-Diligence</b></small></td>
</tr>
<tr>
<td colspan="3" align="center"><img src="wrath-patience/layer=15_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=7b.png" width="33%"></td>
</tr>
<tr>
<td colspan="3" align="center"><small><b>Wrath-Patience</b></small></td>
</tr>
</table>

### Layer 16
<table>
<tr>
<td width="33%"><img src="envy-kindness/layer=16_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="gluttony-temperance/layer=16_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="greed-charity/layer=16_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Envy-Kindness</b></small></td>
<td align="center"><small><b>Gluttony-Temperance</b></small></td>
<td align="center"><small><b>Greed-Charity</b></small></td>
</tr>
<tr>
<td width="33%"><img src="lust-chastity/layer=16_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="pride-humility/layer=16_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="sloth-diligence/layer=16_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Lust-Chastity</b></small></td>
<td align="center"><small><b>Pride-Humility</b></small></td>
<td align="center"><small><b>Sloth-Diligence</b></small></td>
</tr>
<tr>
<td colspan="3" align="center"><img src="wrath-patience/layer=16_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=7b.png" width="33%"></td>
</tr>
<tr>
<td colspan="3" align="center"><small><b>Wrath-Patience</b></small></td>
</tr>
</table>

### Layer 17
<table>
<tr>
<td width="33%"><img src="envy-kindness/layer=17_behavior=envy-kindness_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="gluttony-temperance/layer=17_behavior=gluttony-temperance_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="greed-charity/layer=17_behavior=greed-charity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Envy-Kindness</b></small></td>
<td align="center"><small><b>Gluttony-Temperance</b></small></td>
<td align="center"><small><b>Greed-Charity</b></small></td>
</tr>
<tr>
<td width="33%"><img src="lust-chastity/layer=17_behavior=lust-chastity_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="pride-humility/layer=17_behavior=pride-humility_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
<td width="33%"><img src="sloth-diligence/layer=17_behavior=sloth-diligence_type=open_ended_use_base_model=False_model_size=7b.png" width="100%"></td>
</tr>
<tr>
<td align="center"><small><b>Lust-Chastity</b></small></td>
<td align="center"><small><b>Pride-Humility</b></small></td>
<td align="center"><small><b>Sloth-Diligence</b></small></td>
</tr>
<tr>
<td colspan="3" align="center"><img src="wrath-patience/layer=17_behavior=wrath-patience_type=open_ended_use_base_model=False_model_size=7b.png" width="33%"></td>
</tr>
<tr>
<td colspan="3" align="center"><small><b>Wrath-Patience</b></small></td>
</tr>
</table>

---

## 5. Key Findings

### 5.1 Critical Layer Boundary at Layer 16-17

**All behaviors show dramatic positive steering collapse after layer 16**:
- Layers 0-16: Positive steering effective (40-65%)
- Layers 17-30: Positive steering minimal (<10%)
- Negative steering remains robust across all 31 layers

### 5.2 Bimodal Steering Windows

Most behaviors show two effectiveness windows for positive steering:
1. **Early layers (0-2)**: Strong steering (55-65%)
2. **Mid layers (9-16)**: Sustained steering (45-60%)
3. **Late layers (17-30)**: Steering failure (<10%)

### 5.3 Negative Steering Robustness

Unlike positive steering, negative (vice) steering maintains 40-58% effectiveness across ALL layers, showing no layer-dependent collapse.

### 5.4 Behavior Steering Effectiveness

**Strongest positive steering** (early/mid layers):
1. Wrath-patience: Up to 65% (layers 0-2)
2. Pride-humility: Up to 65% (layer 13)
3. Greed-charity: Up to 64% (layer 13)

---

## 6. Conclusions

The Llama 2 7B Chat model exhibits **extreme layer-dependent steering asymmetry**:

1. **Layer 16-17 boundary**: Sharp transition where positive steering becomes ineffective
2. **Virtue localization**: Chat training concentrated virtue-related representations in layers 0-16
3. **Vice distribution**: Vice representations remain accessible throughout model depth
4. **Optimal intervention**: Layers 9-15 for bidirectional control; layers 0-2 for strong early intervention

**Critical insight**: Chat fine-tuning created a **fragile virtue representation** localized to specific layers, while **vice representations remain robust and distributed**. This asymmetry suggests:
- Alignment training strengthened specific virtue-encoding layers
- Base model vice representations were preserved throughout
- Late-layer processing may bypass or override virtue steering

**Recommendation**: For reliable steering in 7B Chat, restrict interventions to layers 0-16. Steering attempts in layers 17-30 will only affect vice induction, not virtue promotion.
